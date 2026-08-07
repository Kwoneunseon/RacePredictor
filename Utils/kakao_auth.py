"""
카카오톡 토큰 최초 발급 스크립트
한 번만 실행하면 .env에 access_token + refresh_token 저장됨
이후 자동 갱신
"""
import requests
import re
from pathlib import Path

from config import KAKAO_REST_API_KEY

REDIRECT_URI = "http://localhost:8000"
ENV_PATH = Path(__file__).parent.parent / ".env"


def get_auth_url():
    return (
        f"https://kauth.kakao.com/oauth/authorize"
        f"?client_id={KAKAO_REST_API_KEY}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
    )


def fetch_tokens(auth_code: str) -> dict:
    resp = requests.post(
        "https://kauth.kakao.com/oauth/token",
        data={
            "grant_type":   "authorization_code",
            "client_id":    KAKAO_REST_API_KEY,
            "redirect_uri": REDIRECT_URI,
            "code":         auth_code,
        },
    )
    return resp.json()


def save_to_env(access_token: str, refresh_token: str):
    text = ENV_PATH.read_text(encoding="utf-8")

    def replace_or_append(text, key, value):
        pattern = rf"^{key}=.*$"
        new_line = f"{key}={value}"
        if re.search(pattern, text, re.MULTILINE):
            return re.sub(pattern, new_line, text, flags=re.MULTILINE)
        return text.rstrip() + f"\n{new_line}\n"

    text = replace_or_append(text, "KAKAO_ACCESS_TOKEN",  access_token)
    text = replace_or_append(text, "KAKAO_REFRESH_TOKEN", refresh_token)
    ENV_PATH.write_text(text, encoding="utf-8")


def main():
    print("=" * 60)
    print("카카오톡 토큰 발급")
    print("=" * 60)
    print()
    print("1단계: 아래 URL을 브라우저에서 열고 카카오 로그인 후 허용하세요.")
    print()
    print(get_auth_url())
    print()
    print("2단계: 로그인 완료 후 브라우저 주소창이 다음처럼 바뀝니다:")
    print("  https://localhost/?code=XXXXXX...")
    print()
    auth_code = input("code= 뒤의 값을 붙여넣으세요: ").strip()
    if not auth_code:
        print("코드가 비어있습니다. 종료합니다.")
        return

    print("\n토큰 요청 중...")
    result = fetch_tokens(auth_code)

    if "access_token" not in result:
        print(f"실패: {result.get('error_description', result)}")
        return

    access_token  = result["access_token"]
    refresh_token = result.get("refresh_token", "")

    save_to_env(access_token, refresh_token)

    print()
    print("완료! .env에 저장됐습니다.")
    print(f"  KAKAO_ACCESS_TOKEN  = {access_token[:20]}...")
    if refresh_token:
        print(f"  KAKAO_REFRESH_TOKEN = {refresh_token[:20]}...")
    else:
        print("  KAKAO_REFRESH_TOKEN = (미발급 — 카카오 앱 설정에서 refresh_token 허용 필요)")
    print()
    print("테스트 전송을 해볼까요? (y/n) ", end="")
    if input().strip().lower() == "y":
        from Utils.kakaotalk import send_message
        # .env 재로드
        from dotenv import load_dotenv
        load_dotenv(override=True)
        send_message("경마 봇 카카오톡 연결 테스트 완료!")


if __name__ == "__main__":
    main()
