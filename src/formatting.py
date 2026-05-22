import re

def clean_feature_name(name: str) -> str:
    """
    Converts database/dataset column names into beautiful, human-readable titles.
    E.g., 'temperature_c' -> 'Temperature (°C)'
          'rainfall_mm' -> 'Rainfall (mm)'
          'sales_units' -> 'Sales Units'
          'ad_spend_usd' -> 'Ad Spend (USD)'
    """
    if not isinstance(name, str):
        return str(name)
        
    # Check for unit suffixes
    units_map = {
        r'_c$': ' (°C)',
        r'_mm$': ' (mm)',
        r'_usd$': ' (USD)',
        r'_lakhs$': ' (Lakhs)',
        r'_percent$': ' (%)',
        r'_percentage$': ' (%)'
    }
    
    clean_name = name
    suffix_found = ""
    
    for regex, label in units_map.items():
        if re.search(regex, clean_name, re.IGNORECASE):
            clean_name = re.sub(regex, '', clean_name, flags=re.IGNORECASE)
            suffix_found = label
            break
            
    # Replace underscores with spaces and capitalize
    clean_name = clean_name.replace('_', ' ').strip().title()
    
    # Capitalize acronyms correctly
    acronyms = {'Usd': 'USD', 'Id': 'ID', 'Tv': 'TV', 'Co2': 'CO2', 'Vip': 'VIP', 'Ai': 'AI'}
    words = clean_name.split()
    words = [acronyms.get(w, w) for w in words]
    clean_name = " ".join(words)
    
    return clean_name + suffix_found

def clean_ohe_value(val_str: str) -> str:
    """
    Cleans one-hot encoded category columns into beautiful logic statements.
    E.g., 'startup_name_Phitku' -> '**Startup Name** is **Phitku**'
          'product_category_Food' -> '**Product Category** is **Food**'
    """
    if not isinstance(val_str, str):
        return str(val_str)
        
    # Standard prefixes in datasets
    known_prefixes = [
        "product_category", "customer_segment", "day_of_week", 
        "startup_name", "sharks_invested", "industry", "region", "gender"
    ]
    
    for prefix in known_prefixes:
        if val_str.lower().startswith(prefix.lower() + "_"):
            val = val_str[len(prefix) + 1:]
            clean_pref = clean_feature_name(prefix)
            return f"**{clean_pref}** is **{val}**"
            
    # Fallback split
    if "_" in val_str:
        parts = val_str.rsplit("_", 1)
        # If second part is capitalized or numeric, treat as categorical value
        if parts[1] and (parts[1][0].isupper() or parts[1].isdigit() or len(parts[1]) <= 3):
            clean_pref = clean_feature_name(parts[0])
            return f"**{clean_pref}** is **{parts[1]}**"
            
    return f"**{clean_feature_name(val_str)}**"

def md_to_html(text: str) -> str:
    """
    Translates basic markdown bolding (**) and italics (*) into clean HTML tags.
    """
    if not isinstance(text, str):
        return str(text)
    # Convert bold **text** to <strong>text</strong>
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    # Convert italics *text* to <em>text</em>
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    return text

