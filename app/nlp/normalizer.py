import re

class PersianNormalizer:
    def __init__(self):
        # Transliteration mappings
        self.arabic_to_persian_chars = {
            'ي': 'ی',
            'ك': 'ک',
            'ۀ': 'هٔ',
            'ة': 'ه',
        }
        
    def normalize_text(self, text: str) -> str:
        """
        Pure Python normalizer for Persian text. Does not require external C++ compiled packages.
        """
        if not text:
            return ""
            
        # 1. Normalize characters (Arabic to Persian)
        for ar_char, fa_char in self.arabic_to_persian_chars.items():
            text = text.replace(ar_char, fa_char)
            
        # 2. Normalize spacing around punctuation
        # Remove spaces before punctuation
        text = re.sub(r'\s+([\.،؛؟!])', r'\1', text)
        # Ensure space after punctuation (except if it's end of string)
        text = re.sub(r'([\.،؛؟!])([^\s\.،؛؟!])', r'\1 \2', text)
        
        # 3. Handle half-spaces (ZWNJ) for common prefixes/suffixes
        # 'می' prefix (e.g., 'می شود' -> 'می‌شود')
        text = re.sub(r'\bمی\s+(?=\w)', 'می‌', text)
        # 'نمی' prefix
        text = re.sub(r'\bنمی\s+(?=\w)', 'نمی‌', text)
        # 'ها' suffix (e.g., 'کتاب ها' -> 'کتاب‌ها')
        text = re.sub(r'\s+ها\b', '‌ها', text)
        # 'های' suffix
        text = re.sub(r'\s+های\b', '‌های', text)
        # 'تر' and 'ترین' suffixes
        text = re.sub(r'\s+تر(ین)?\b', r'‌تر\1', text)
        
        # 4. Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def process_segment(self, segment_text: str) -> str:
        return self.normalize_text(segment_text)

if __name__ == "__main__":
    n = PersianNormalizer()
    test_cases = [
        "سلام ،   این روزها هوش مصنوعی  خیلی مورد توجه قرار گرفته .",
        "تقریباً هر هفته خبر جدیدی درباره مدل های پیشرفته منتشر می شود .",
        "کتاب ها را بیاور ترجیحا جدید ترین ها را",
        "كیوان به مدرسه می رفت ."
    ]
    for tc in test_cases:
        print(f"Original: {tc}")
        print(f"Normalized: {n.normalize_text(tc)}")
        print("-" * 30)
