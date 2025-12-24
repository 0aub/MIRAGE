
import sys

def test_encoding():
    # Arabic text: "تجربة النص العربي"
    # Windows-1256 bytes
    text = "تجربة النص العربي"
    try:
        bytes_1256 = text.encode("windows-1256")
    except:
        print("Could not encode to windows-1256")
        return

    print(f"Original: {text}")
    print(f"Bytes (1256): {bytes_1256}")

    # Case 1: Read as UTF-8 (Strict) -> Error
    try:
        decoded = bytes_1256.decode("utf-8")
        print(f"Decoded UTF-8: {decoded}")
    except Exception as e:
        print(f"Decoded UTF-8 (Strict): Error - {e}")

    # Case 2: Read as UTF-8 (Ignore) -> Lossy
    try:
        decoded = bytes_1256.decode("utf-8", errors="ignore")
        print(f"Decoded UTF-8 (Ignore): {decoded}")
    except Exception as e:
        print(f"Decoded UTF-8 (Ignore): Error - {e}")
        
    # Case 3: Read as UTF-8 (Replace) -> 
    try:
        decoded = bytes_1256.decode("utf-8", errors="replace")
        print(f"Decoded UTF-8 (Replace): {decoded}")
    except Exception as e:
        print(f"Decoded UTF-8 (Replace): Error - {e}")

    # Case 4: Read as Latin-1 (Windows-1252) -> Mojibake "Š ÅÅ¼"
    try:
        decoded = bytes_1256.decode("windows-1252")
        print(f"Decoded Windows-1252 (Mojibake): {decoded}")
        
        # Test specific chars mentioned by user
        # User saw: Š ÅÅ¼
        # Š = 0x8A (1252) -> ٹ (1256)? No, 0x8A is 138.
        # 1256 mapping: 0x8A is ٹ (U+0679) ? 
        # Actually standard 1256: 0x8A is ٹ (U+0679) Tteh.
        
        # Let's check "خ" (Kha). 
        # utf-8: D8 Ae 
        # 1256: Ce (206)
        # 1252: C5 is Å. C6 is Æ.
        # So 0xC5 in 1256 is خ?
        # 0xC5 (197) is خ (U+062E) in Windows-1256.
        # 0xC5 (197) is Å (U+00C5) in Windows-1252.
        
        # YES! This confirms the user is seeing 1256 bytes interpreted as 1252.
        
    except Exception as e:
        print(f"Decoded Windows-1252: Error - {e}")

if __name__ == "__main__":
    test_encoding()
