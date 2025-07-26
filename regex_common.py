# core/utils/regex_common.py
import re
import logging

logger = logging.getLogger(__name__)

# --- DOC NUMBER REGEX ---
DOC_NUMBER_PATTERNS = [
    r"\bRA[CĆ]UN\s*(br\.?|broj\.?|[-:\s])?\s*([A-Z0-9\-\/]+)",
    r"\bR1\s*RA[CĆ]UN.*?br\.?\s*([A-Z0-9\-\/]+)",
    r"\bOTP.?BR\.?\s*([A-Z0-9\-\/]+)",
    r"\bBROJ\s*[:\-]?\s*([A-Z0-9\-\/]+)",
    r"\bINVOICE\s*(NO\.?|#)?[:\s]*([A-Z0-9\-\/]+)",
    r"\bRA[CĆ]UN[- ]?OTPREMNICA\s*([A-Z0-9\-\/]+)",
    r"\bRA[CĆ]UN([A-Z0-9\-\/]+)",
    r"\bINV(OICE)?[-\s]*#?\s*([A-Z0-9\-\/]+)",
    r"\bFAKTURA\s*#?\s*([A-Z0-9\-\/]+)",
    r"\b([A-Z]{2,4}-[0-9]{3,6}[-\/]?[0-9]{0,6})\b",
    r"\b([0-9]{3,6}\/[A-Z0-9]{2,6}\/[0-9]{1,6})\b",
    r"\b([0-9]{6,})\b",
]

OIB_PATTERN = r"\b[0-9]{11}\b"

COUNTRY_VAT_REGEX = {
    "EU": r"EU\d{8,12}",
    "AT": r"ATU\d{8}",
    "BE": r"BE0\d{9}",
    "BG": r"BG\d{9,10}",
    "CY": r"CY\d{8}[A-Z]",
    "CZ": r"CZ\d{8,10}",
    "DK": r"DK\d{8}",
    "EE": r"EE\d{9}",
    "FI": r"FI\d{8}",
    "FR": r"FR[A-Z0-9]{2}\d{9}",
    "EL": r"EL\d{9}",
    "HR": r"HR\d{11}",
    "IE": r"IE\d{7}[A-W]|IE\d[A-Z0-9]\d{5}[A-W]",
    "IT": r"IT\d{11}",
    "LV": r"LV\d{11}",
    "LT": r"LT(\d{9}|\d{12})",
    "LU": r"LU\d{8}",
    "HU": r"HU\d{8}",
    "MT": r"MT\d{8}",
    "NL": r"NL\d{9}B\d{2}",
    "DE": r"DE\d{9}",
    "PL": r"PL\d{10}",
    "PT": r"PT\d{9}",
    "RO": r"RO\d{2,10}",
    "SK": r"SK\d{10}",
    "SI": r"SI\d{8}",
    "ES": r"ES[A-Z0-9]\d{7}[A-Z0-9]",
    "SE": r"SE\d{12}",
}

VAT_CANDIDATE_PATTERN = r"\b(EU|AT|BE|BG|CY|CZ|DK|EE|FI|FR|EL|HR|IE|IT|LV|LT|LU|HU|MT|NL|DE|PL|PT|RO|SK|SI|ES|SE)\s*(\d+)\s*([A-Z]?)\b"

DATE_PATTERNS = [
    (r"(vrijeme izdavanja|datum izdavanja|račun izdan|datum računa|datum i mjesto|vrijeme i mjesto izdavanja)[:\s]*([0-9]{1,2}\.[0-9]{1,2}\.[0-9]{4})", "invoice_date"),
    (r"(rok pla[cć]anja|datum dospijeća|valuta|datum valute)[:\s]*([0-9]{1,2}\.[0-9]{1,2}\.[0-9]{4})", "due_date"),
    (r"(datum isporuke|isporuka)[:\s]*([0-9]{1,2}\.[0-9]{1,2}\.[0-9]{4})", "delivery_date"),
    (r"(datum unosa)[:\s]*([0-9]{1,2}\.[0-9]{1,2}\.[0-9]{4})", "entry_date"),
    (r"\b([0-9]{1,2}\.[0-9]{1,2}\.[0-9]{4})\b", "any_date"),
]

def extract_doc_number(text: str) -> str | None:
    for pattern in DOC_NUMBER_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # grupa koja sadrži broj dokumenta je zadnja grupa (group 2 u većini)
            group_index = 2 if match.lastindex and match.lastindex >= 2 else 0
            return match.group(group_index).strip()
    return None

def extract_oib(text: str) -> str | None:
    hr_oib_match = re.search(r"HR(\d{11})", text, re.IGNORECASE)
    if hr_oib_match:
        return hr_oib_match.group(1)

    matches = re.findall(OIB_PATTERN, text)
    if not matches:
        return None
    return matches[0]

def extract_all_oibs(text: str) -> list[str]:
    return re.findall(OIB_PATTERN, text)

def extract_vat_number(text: str) -> str | None:
    candidates = re.findall(VAT_CANDIDATE_PATTERN, text.upper())
    logger.debug(f"Pronađeni kandidati za VAT broj (raw): {candidates}")

    for country_code, number, suffix in candidates:
        candidate = f"{country_code}{number}{suffix}"
        pattern = COUNTRY_VAT_REGEX.get(country_code)

        if not pattern:
            logger.debug(f"Nema pravila za zemlju {country_code} za VAT broj {candidate}, preskačem.")
            continue

        if re.fullmatch(pattern, candidate):
            logger.debug(f"Validan VAT broj pronađen: {candidate}")
            return candidate
        else:
            logger.debug(f"Nevalidan VAT broj za zemlju {country_code}: {candidate}")

    return None

def extract_all_vats(text: str) -> list[str]:
    raw_candidates = re.findall(VAT_CANDIDATE_PATTERN, text.upper())
    vats = []
    for country_code, number, suffix in raw_candidates:
        candidate = f"{country_code}{number}{suffix}"
        regex = COUNTRY_VAT_REGEX.get(country_code)
        if regex and re.fullmatch(regex, candidate):
            vats.append(candidate)
    return vats

def extract_dates(text: str) -> dict:
    result = {}
    for pat, label in DATE_PATTERNS:
        for match in re.finditer(pat, text, re.IGNORECASE):
            value = match.group(2 if match.lastindex and match.lastindex > 1 else 1)
            if label not in result:
                result[label] = value
    return result

def extract_invoice_date(text: str) -> str | None:
    dates = extract_dates(text)
    return dates.get("invoice_date")

def extract_due_date(text: str) -> str | None:
    dates = extract_dates(text)
    return dates.get("due_date")
