import os
from datetime import datetime
from typing import Optional, List

class TxtWriter:
    """TXT 파일을 생성하고 관리하는 클래스"""
    
    def __init__(self, output_dir: str = "output", auto_timestamp: bool = True):
        """
        Args:
            output_dir: 저장할 디렉토리 경로
            auto_timestamp: 파일명에 자동으로 타임스탬프 추가 여부
        """
        self.output_dir = output_dir
        self.auto_timestamp = auto_timestamp
        self.lines: List[str] = []
        self.current_file_path: Optional[str] = None
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
    
    def write_line(self, text: str) -> None:
        """한 줄 추가"""
        self.lines.append(str(text))
    
    def write_lines(self, texts: List[str]) -> None:
        """여러 줄 추가"""
        for text in texts:
            self.lines.append(str(text))
    
    def write_separator(self, char: str = "-", length: int = 50) -> None:
        """구분선 추가"""
        self.lines.append(char * length)
    
    def write_header(self, title: str, char: str = "=", length: int = 60) -> None:
        """헤더 추가"""
        self.lines.append(char * length)
        self.lines.append(title)
        self.lines.append(char * length)
    
    def write_blank_line(self, count: int = 1) -> None:
        """빈 줄 추가"""
        for _ in range(count):
            self.lines.append("")
    
    def clear(self) -> None:
        """저장된 내용 모두 삭제"""
        self.lines.clear()
    
    def preview(self, max_lines: int = 20) -> None:
        """저장될 내용 미리보기"""
        print("📋 저장될 내용 미리보기:")
        print("-" * 40)
        
        for i, line in enumerate(self.lines[:max_lines]):
            print(f"{i+1:3d}: {line}")
        
        if len(self.lines) > max_lines:
            print(f"... (총 {len(self.lines)}줄 중 {max_lines}줄만 표시)")
        print("-" * 40)
    
    def save(self, filename: Optional[str] = None, encoding: str = "euc-kr") -> str:
        """
        파일로 저장
        
        Args:
            filename: 파일명 (None이면 자동 생성)
            encoding: 인코딩 방식
            
        Returns:
            저장된 파일의 전체 경로
        """
        if filename is None:
            if self.auto_timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"output_{timestamp}.txt"
            else:
                filename = "output.txt"
        
        # .txt 확장자 자동 추가
        if not filename.endswith('.txt'):
            filename += '.txt'
        
        file_path = os.path.join(self.output_dir, filename)
        
        try:
            with open(file_path, 'w', encoding=encoding) as f:
                f.write('\n'.join(self.lines))
            
            self.current_file_path = file_path
            print(f"💾 파일 저장 완료: {file_path}")
            print(f"📊 총 {len(self.lines)}줄 저장됨")
            return file_path
            
        except Exception as e:
            print(f"❌ 파일 저장 실패: {e}")
            raise
    
    def append_to_existing(self, file_path: str, encoding: str = "euc-kr") -> None:
        """기존 파일에 내용 추가"""
        try:
            with open(file_path, 'a', encoding=encoding) as f:
                f.write('\n' + '\n'.join(self.lines))
            print(f"📝 파일에 내용 추가 완료: {file_path}")
        except Exception as e:
            print(f"❌ 파일 추가 실패: {e}")
            raise
    
    def get_line_count(self) -> int:
        """현재 저장된 줄 수 반환"""
        return len(self.lines)
    
    def get_content(self) -> str:
        """현재 저장된 내용을 문자열로 반환"""
        return '\n'.join(self.lines)