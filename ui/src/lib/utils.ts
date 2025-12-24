import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Helper to fix encoding issues
export const fixEncoding = (text: string): string => {
  if (!text) return "";

  // 1. If it already looks like Arabic (UTF-8), return it
  if (/[\u0600-\u06FF]/.test(text)) return text;

  // 2. Check for ASMO-449 (7-bit Arabic)
  const asmoMap: Record<string, string> = {
    '!': 'ء', '"': 'آ', '#': 'أ', '$': 'ؤ', '%': 'إ', '&': 'ئ', "'": 'ا', '(': 'ب',
    ')': 'ة', '*': 'ت', '+': 'ث', ',': 'ج', '-': 'ح', '.': 'خ', '/': 'د', '0': 'ذ',
    '1': 'ر', '2': 'ز', '3': 'س', '4': 'ش', '5': 'ص', '6': 'ض', '7': 'ط', '8': 'ظ',
    '9': 'ع', ':': 'غ', '<': 'ك', '=': 'ل', '>': 'م', '?': 'ن', '@': 'ه',
    'A': 'ف', 'B': 'ق', 'C': 'ك', 'D': 'ل', 'E': 'م', 'F': 'ن', 'G': 'ه', 'H': 'و', 'I': 'ى', 'J': 'ي',
    'K': 'ً', 'L': 'ٌ', 'M': 'ٍ', 'N': 'َ', 'O': 'ُ', 'P': 'ِ', 'Q': 'ّ', 'R': 'ْ'
  };

  if (!/[a-z]/.test(text) && /[A-J0-9!'(]/.test(text)) {
      let asmoDecoded = "";
      for (const char of text) {
          if (asmoMap[char]) {
              asmoDecoded += asmoMap[char];
          } else if (char === ' ' || char === '\n' || char === '\r' || char === '\t') {
              asmoDecoded += char;
          } else if (/[\[\]_`^{}|~]/.test(char)) {
             asmoDecoded += char;
          } else {
             asmoDecoded += char;
          }
      }
      if (/[\u0600-\u06FF]/.test(asmoDecoded)) {
          return asmoDecoded;
      }
  }

  // 3. Check for Windows-1256 restoration (reversing 1252/1255)
  const bytes = new Uint8Array(text.length);
  const win1252Reversal: Record<number, number> = {
    0x20AC: 0x80, 0x201A: 0x82, 0x0192: 0x83, 0x201E: 0x84, 0x2026: 0x85, 0x2020: 0x86, 0x2021: 0x87, 0x02C6: 0x88, 0x2030: 0x89, 0x0160: 0x8A, 0x2039: 0x8B, 0x0152: 0x8C, 0x017D: 0x8E,
    0x2018: 0x91, 0x2019: 0x92, 0x201C: 0x93, 0x201D: 0x94, 0x2022: 0x95, 0x2013: 0x96, 0x2014: 0x97, 0x02DC: 0x98, 0x2122: 0x99, 0x0161: 0x9A, 0x203A: 0x9B, 0x0153: 0x9C, 0x017E: 0x9E, 0x0178: 0x9F
  };
  const hebrewOffset = 0x05D0 - 0xE0; 

  let hasHighChars = false;

  for (let i = 0; i < text.length; i++) {
      const code = text.charCodeAt(i);
      if (code < 256) {
          bytes[i] = code;
          if (code > 127) hasHighChars = true;
      } else if (win1252Reversal[code]) {
          bytes[i] = win1252Reversal[code];
          hasHighChars = true;
      } else if (code >= 0x05D0 && code <= 0x05EA) { 
          bytes[i] = code - hebrewOffset;
          hasHighChars = true;
      } else {
          bytes[i] = code & 0xFF;
      }
  }

  try {
      const decoder = new TextDecoder("utf-8", { fatal: true });
      const decoded = decoder.decode(bytes);
      if (/[\u0600-\u06FF]/.test(decoded)) return decoded;
  } catch (e) {}

  if (hasHighChars) {
    try {
        const decoder = new TextDecoder("windows-1256");
        const decoded = decoder.decode(bytes);
        if (/[\u0600-\u06FF]/.test(decoded)) return decoded;
    } catch (e) {}
  }

  try {
      return decodeURIComponent(escape(text));
  } catch (e) {
      return text;
  }
};
