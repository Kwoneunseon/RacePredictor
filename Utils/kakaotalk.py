import requests
import json
import webbrowser
from datetime import datetime


class KakaoSender:
    def __init__(self, rest_api_key):
        self.rest_api_key = rest_api_key
        self.redirect_uri = "http://localhost:8000"
        self.access_token = None
        self._authenticated = False
    
    def authenticate(self):
        """카카오 인증 (최초 1회만 실행)"""
        if self._authenticated:
            return True
        
        print("🔐 카카오톡 인증이 필요합니다...")
        
        # 인증 URL 생성
        auth_url = (
            f"https://kauth.kakao.com/oauth/authorize?"
            f"client_id={self.rest_api_key}&"
            f"redirect_uri={self.redirect_uri}&"
            f"response_type=code"
        )
        
        print(f"인증 URL: {auth_url}")
        
        try:
            webbrowser.open(auth_url)
            print("✅ 브라우저가 열렸습니다. 카카오 로그인 후 인증 코드를 입력하세요.")
        except:
            print("❌ 브라우저 자동 열기 실패. 위 URL을 복사해서 브라우저에 붙여넣으세요.")
        
        # 인증 코드 입력
        auth_code = input("\n인증 코드 입력: ").strip()
        
        # 액세스 토큰 요청
        if self._get_access_token(auth_code):
            self._authenticated = True
            print("✅ 인증 완료!")
            return True
        
        return False
    
    def _get_access_token(self, auth_code):
        """액세스 토큰 발급"""
        token_url = "https://kauth.kakao.com/oauth/token"
        data = {
            'grant_type': 'authorization_code',
            'client_id': self.rest_api_key,
            'redirect_uri': self.redirect_uri,
            'code': auth_code
        }
        
        try:
            response = requests.post(token_url, data=data)
            token_info = response.json()
            
            if 'access_token' in token_info:
                self.access_token = token_info['access_token']
                return True
            else:
                print(f"❌ 토큰 발급 실패: {token_info.get('error_description', '알 수 없는 오류')}")
                return False
        
        except Exception as e:
            print(f"❌ 네트워크 오류: {e}")
            return False
    
    def send(self, message):
        """메시지 전송 (메인 기능)"""
        # 인증 확인
        if not self._authenticated:
            if not self.authenticate():
                return False
        
        # 메시지 전송
        return self._send_message(message)
    
    def _send_message(self, message):
        """실제 메시지 전송"""
        url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        # 현재 시간 포함
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"📱 {message}\n\n⏰ {timestamp}"
        
        template_object = {
            "object_type": "text",
            "text": formatted_message
        }
        
        data = {
            "template_object": json.dumps(template_object)
        }
        
        try:
            response = requests.post(url, headers=headers, data=data)
            result = response.json()
            
            if response.status_code == 200:
                print(f"✅ 메시지 전송 완료: {message}")
                return True
            else:
                error_msg = result.get('msg', '알 수 없는 오류')
                print(f"❌ 전송 실패: {error_msg}")
                
                # 토큰 만료 시 재인증
                if 'token' in error_msg.lower() or result.get('code') == -401:
                    print("🔄 토큰이 만료되었습니다. 재인증을 시도합니다...")
                    self._authenticated = False
                    return self.send(message)
                
                return False
        
        except Exception as e:
            print(f"❌ 네트워크 오류: {e}")
            return False

# ===========================================
# 사용 방법
# ===========================================

# 1. API 키 설정 (config에서 로드)
from config import KAKAO_REST_API_KEY, KAKAO_ACCESS_TOKEN

# 2. KakaoSender 인스턴스 생성
kakao = KakaoSender(KAKAO_REST_API_KEY)
if KAKAO_ACCESS_TOKEN:
    kakao.access_token = KAKAO_ACCESS_TOKEN
    kakao._authenticated = True

# 3. 메시지 전송 함수들
def send_message(text):
    """간단한 메시지 전송"""
    return kakao.send(text)

def send_notification(title, content):
    """알림 형태로 전송"""
    message = f"🔔 {title}\n\n{content}"
    return kakao.send(message)

def send_status(status):
    """상태 메시지 전송"""
    message = f"📊 상태 업데이트\n\n{status}"
    return kakao.send(message)

# ===========================================
# 사용 예시
# ===========================================

if __name__ == "__main__":
    # 기본 사용법
    send_message("안녕하세요! 테스트 메시지입니다.")
    
    # 알림 전송
    send_notification("작업 완료", "데이터 처리가 완료되었습니다.")
    
    # 상태 전송
    send_status("서버가 정상적으로 실행 중입니다.")
    
    # 사용자 입력받아서 전송
    while True:
        user_input = input("\n전송할 메시지 (종료: quit): ")
        if user_input.lower() == 'quit':
            break
        send_message(user_input)