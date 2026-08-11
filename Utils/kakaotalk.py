"""
카카오톡 메시지 전송 유틸리티
- access_token 만료 시 refresh_token으로 자동 갱신
- .env 파일에 토큰 자동 업데이트
"""
import os
import re
import json
import requests
from pathlib import Path
from Utils.logger import get_logger

logger = get_logger(__name__)

ENV_PATH   = Path(__file__).parent.parent / ".env"
REDIRECT_URI = "http://localhost:8000"


def _load_env_value(key: str) -> str:
    text = ENV_PATH.read_text(encoding="utf-8")
    m = re.search(rf"^{key}=(.*)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _save_env_value(key: str, value: str):
    text = ENV_PATH.read_text(encoding="utf-8")
    pattern = rf"^{key}=.*$"
    new_line = f"{key}={value}"
    if re.search(pattern, text, re.MULTILINE):
        text = re.sub(pattern, new_line, text, flags=re.MULTILINE)
    else:
        text = text.rstrip() + f"\n{new_line}\n"
    ENV_PATH.write_text(text, encoding="utf-8")
    # 런타임 환경변수도 갱신
    os.environ[key] = value


def _refresh_access_token(rest_api_key: str, refresh_token: str) -> str | None:
    """refresh_token으로 새 access_token 발급, .env에 저장"""
    resp = requests.post(
        "https://kauth.kakao.com/oauth/token",
        data={
            "grant_type":    "refresh_token",
            "client_id":     rest_api_key,
            "refresh_token": refresh_token,
        },
        timeout=10,
    )
    result = resp.json()
    if "access_token" not in result:
        logger.error(f"카카오 토큰 갱신 실패: {result}")
        return None

    new_access = result["access_token"]
    _save_env_value("KAKAO_ACCESS_TOKEN", new_access)
    logger.info("카카오 access_token 갱신 완료")

    # refresh_token도 새로 발급된 경우 저장
    if "refresh_token" in result:
        _save_env_value("KAKAO_REFRESH_TOKEN", result["refresh_token"])

    return new_access


def send_message(text: str) -> bool:
    """
    카카오톡 나에게 메시지 전송.
    access_token 만료 시 refresh_token으로 1회 자동 갱신 후 재시도.
    """
    rest_api_key  = os.getenv("KAKAO_REST_API_KEY", "")
    access_token  = _load_env_value("KAKAO_ACCESS_TOKEN")
    refresh_token = _load_env_value("KAKAO_REFRESH_TOKEN")

    if not access_token:
        logger.error("KAKAO_ACCESS_TOKEN이 비어있습니다. `python -m scripts.kakao_auth`를 먼저 실행하세요.")
        return False

    def _do_send(token: str) -> requests.Response:
        return requests.post(
            "https://kapi.kakao.com/v2/api/talk/memo/default/send",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type":  "application/x-www-form-urlencoded",
            },
            data={
                "template_object": json.dumps({
                    "object_type": "text",
                    "text":        text[:1999],   # 카카오 최대 2000자
                    "link":        {"web_url": "", "mobile_web_url": ""},
                })
            },
            timeout=10,
        )

    resp = _do_send(access_token)

    # 토큰 만료(-401) 또는 인증 오류 → refresh 후 재시도
    if resp.status_code != 200:
        result = resp.json()
        code = result.get("code", 0)
        if code == -401 and refresh_token:
            logger.info("access_token 만료 — 자동 갱신 시도")
            new_token = _refresh_access_token(rest_api_key, refresh_token)
            if new_token:
                resp = _do_send(new_token)
            else:
                logger.error("토큰 갱신 실패. `python -m scripts.kakao_auth`를 다시 실행하세요.")
                return False
        else:
            logger.error(f"카카오톡 전송 실패 (code={code}): {result.get('msg', '')}")
            return False

    if resp.status_code == 200:
        logger.info("카카오톡 전송 완료")
        return True

    logger.error(f"카카오톡 전송 실패: {resp.status_code} {resp.text[:100]}")
    return False


def send_notification(title: str, content: str) -> bool:
    return send_message(f"[{title}]\n\n{content}")
