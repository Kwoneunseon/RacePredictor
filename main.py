import argparse
from datetime import datetime

from config import MODEL_DIR
from Utils.database import get_client, test_connection
from Utils.logger import get_logger
from core.preprocessor import RacePreprocessor
from core.trainer import RaceModelTrainer
from core.predictor import RacePredictor
from core.model_manager import ModelManager
from core.investor import Investor

logger = get_logger(__name__)


def cmd_train(args):
    logger.info("=== 모델 학습 시작 ===")
    import pandas as pd
    client = get_client()

    result = client.table("race_entries").select(
        "*, horses(*), races(*), jockeys(*), trainers(*)"
    ).gte("race_date", args.start_date).lte("race_date", args.end_date)\
        .not_.is_("final_rank", "null").execute()

    if not result.data:
        logger.error("학습 데이터 없음")
        return

    df = pd.DataFrame(result.data)
    logger.info(f"학습 데이터: {len(df)}건")

    preprocessor = RacePreprocessor()
    df_processed = preprocessor.fit_transform(df)

    trainer = RaceModelTrainer()
    metrics = trainer.train(df_processed)

    mm = ModelManager()
    version = mm.save(trainer, preprocessor, metrics=metrics["metrics"])
    logger.info(f"모델 저장 완료: {version}")


def cmd_predict(args):
    logger.info(f"=== 예측 시작: {args.date} ===")

    preprocessor = RacePreprocessor()
    trainer = RaceModelTrainer()
    mm = ModelManager()

    if not mm.load(trainer, preprocessor):
        logger.error("로드할 모델 없음. 먼저 --mode train 실행")
        return

    predictor = RacePredictor(trainer, preprocessor)
    predictions = predictor.predict_from_db(args.date, meet_code=args.meet)

    if predictions.empty:
        logger.warning(f"{args.date} 예측 결과 없음")
        return

    print("\n=== 예측 결과 ===")
    for _, row in predictions.head(3).iterrows():
        print(f"  {int(row['predicted_rank'])}위 예측: "
              f"{row.get('horse_name', row.get('horse_id', '?'))} "
              f"(확률: {row['win_probability']:.3f})")

    investor = Investor(threshold=0.6, bet_amount=1000)
    bets = investor.decide(predictions)
    if not bets.empty:
        print(f"\n베팅 추천: {len(bets)}건")
        for _, bet in bets.iterrows():
            print(f"  → {bet.get('horse_name', bet.get('horse_id', '?'))} "
                  f"(확률: {bet['win_probability']:.3f})")


def cmd_status(args):
    print("=== 시스템 상태 ===")
    connected = test_connection()
    print(f"DB 연결: {'OK' if connected else 'FAIL'}")
    mm = ModelManager()
    models = list(MODEL_DIR.glob("model_*.pkl"))
    print(f"저장된 모델: {len(models)}개")
    if models:
        print(f"최신 모델: {sorted(models)[-1].name}")
    print(f"재학습 필요: {'YES' if mm.is_outdated() else 'NO'}")


def main():
    parser = argparse.ArgumentParser(description="경마 자동 투자 봇")
    parser.add_argument("--mode", choices=["train", "predict", "status"],
                        default="status")
    parser.add_argument("--start-date", default="2024-01-01",
                        dest="start_date")
    parser.add_argument("--end-date",
                        default=datetime.now().strftime("%Y-%m-%d"),
                        dest="end_date")
    parser.add_argument("--date",
                        default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--meet", type=int, default=None,
                        help="1=서울, 2=부산, 3=제주")
    args = parser.parse_args()

    if args.mode == "train":
        cmd_train(args)
    elif args.mode == "predict":
        cmd_predict(args)
    elif args.mode == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
