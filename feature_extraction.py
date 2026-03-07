import pandas as pd
import re

# Load new dataset with 'Label' column and 'good'/'bad' values
df = pd.read_csv("phishing_site_urls.csv")

def extract_features(url):
    features = {}
    
    # Lexical features
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_slashes'] = url.count('/')
    features['num_digits'] = sum(c.isdigit() for c in url)
    
    # Subdomain count
    domain_part = re.findall(r'://([^/]+)', url)
    if domain_part:
        domain_split = domain_part[0].split('.')
        features['num_subdomains'] = max(len(domain_split) - 2, 0)
    else:
        features['num_subdomains'] = 0
    
    # Check if IP address in URL
    ip_pattern = re.compile(r'\b\d{1,3}(?:\.\d{1,3}){3}\b')
    features['has_ip'] = 1 if ip_pattern.search(url) else 0
    
    # Keywords
    suspicious_words = ['login','verify','update','secure','bank','free','account','click','confirm','reset']
    for word in suspicious_words:
        features[f'word_{word}'] = 1 if word in url.lower() else 0
    
    # Protocol & short URL
    features['https'] = 1 if url.startswith('https://') else 0
    shorteners = ['bit.ly','tinyurl','goo.gl','ow.ly']
    features['short_url'] = 1 if any(s in url for s in shorteners) else 0
    
    return features


# Apply to dataframe
features_df = df['URL'].apply(lambda x: pd.Series(extract_features(x)))
final_df = pd.concat([features_df, df['Label']], axis=1)

# Save features to CSV
final_df.to_csv('phishing_features.csv', index=False)

print("Features saved to phishing_features.csv")
print(final_df.head())
