from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import joblib
import os

# Path to the trained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'random_forest_model.joblib')

model = joblib.load(MODEL_PATH)

# ...existing code...

# Feature extraction based on feature_extraction.py
import re
def extract_features(url):
	features = {}
	features['url_length'] = len(url)
	features['num_dots'] = url.count('.')
	features['num_hyphens'] = url.count('-')
	features['num_slashes'] = url.count('/')
	features['num_digits'] = sum(c.isdigit() for c in url)

	domain_part = re.findall(r'://([^/]+)', url)
	if domain_part:
		domain_split = domain_part[0].split('.')
		features['num_subdomains'] = max(len(domain_split) - 2, 0)
	else:
		features['num_subdomains'] = 0

	ip_pattern = re.compile(r'\b\d{1,3}(?:\.\d{1,3}){3}\b')
	features['has_ip'] = 1 if ip_pattern.search(url) else 0
	features['has_at_symbol'] = 1 if '@' in url else 0

	suspicious_words = ['login','verify','update','secure','bank','free','account','click','confirm','reset']
	features['suspicious_keywords'] = sum(1 for word in suspicious_words if word in url.lower())
	for word in suspicious_words:
		features[f'word_{word}'] = 1 if word in url.lower() else 0

	features['https'] = 1 if url.startswith('https://') else 0
	shorteners = ['bit.ly','tinyurl','goo.gl','ow.ly']
	features['short_url'] = 1 if any(s in url for s in shorteners) else 0

	# Ensure feature order matches model training
	feature_order = [
		'url_length', 'dot_count', 'num_hyphens', 'num_slashes', 'num_digits',
		'num_subdomains', 'has_ip_address',
		'word_login', 'word_verify', 'word_update', 'word_secure', 'word_bank',
		'word_free', 'word_account', 'word_click', 'word_confirm', 'word_reset',
		'https', 'short_url'
	]
	model_features = [features.get(f, 0) for f in feature_order]
	return model_features, features

# Debug: print model classes and sample feature vector for google.com
print('Model classes:', model.classes_)
def debug_features():
	url = 'https://www.google.com'
	features, _ = extract_features(url)
	print('Features for google.com:', features)
debug_features()


# Feature extraction based on feature_extraction.py
import re
def extract_features(url):
	features = {}
	features['url_length'] = len(url)
	features['dot_count'] = url.count('.')
	features['num_hyphens'] = url.count('-')
	features['num_slashes'] = url.count('/')
	features['num_digits'] = sum(c.isdigit() for c in url)

	domain_part = re.findall(r'://([^/]+)', url)
	if domain_part:
		domain_split = domain_part[0].split('.')
		features['num_subdomains'] = max(len(domain_split) - 2, 0)
	else:
		features['num_subdomains'] = 0

	ip_pattern = re.compile(r'\b\d{1,3}(?:\.\d{1,3}){3}\b')
	features['has_ip_address'] = 1 if ip_pattern.search(url) else 0
	features['has_at_symbol'] = 1 if '@' in url else 0

	suspicious_words = ['login','verify','update','secure','bank','free','account','click','confirm','reset']
	features['suspicious_keywords'] = sum(1 for word in suspicious_words if word in url.lower())
	for word in suspicious_words:
		features[f'word_{word}'] = 1 if word in url.lower() else 0

	features['https'] = 1 if url.startswith('https://') else 0
	shorteners = ['bit.ly','tinyurl','goo.gl','ow.ly']
	features['short_url'] = 1 if any(s in url for s in shorteners) else 0

	# Ensure feature order matches model training
	feature_order = [
		'url_length', 'dot_count', 'num_hyphens', 'num_slashes', 'num_digits',
		'num_subdomains', 'has_ip_address',
		'word_login', 'word_verify', 'word_update', 'word_secure', 'word_bank',
		'word_free', 'word_account', 'word_click', 'word_confirm', 'word_reset',
		'https', 'short_url'
	]
	model_features = [features.get(f, 0) for f in feature_order]
	return model_features, features

@csrf_exempt
def predict(request):
	if request.method == 'POST':
		try:
			data = json.loads(request.body)
			url = data.get('url')
			if not url:
				return JsonResponse({'error': 'No URL provided'}, status=400)
			import pandas as pd
			model_features, feature_dict = extract_features(url)
			feature_names = [
				'url_length', 'num_dots', 'num_hyphens', 'num_slashes', 'num_digits',
				'num_subdomains', 'has_ip',
				'word_login', 'word_verify', 'word_update', 'word_secure', 'word_bank',
				'word_free', 'word_account', 'word_click', 'word_confirm', 'word_reset',
				'https', 'short_url'
			]
			X = pd.DataFrame([model_features], columns=feature_names)
			prediction_arr = model.predict(X)
			proba_arr = model.predict_proba(X)
			print('model.classes_:', model.classes_)
			print('model.predict(X):', prediction_arr)
			print('model.predict_proba(X):', proba_arr)
			prediction = prediction_arr[0]
			# Dynamically find index for 'benign' (safe) class
			safe_label = 'benign'
			if safe_label in model.classes_:
				safe_index = list(model.classes_).index(safe_label)
				safe_proba = proba_arr[0][safe_index]
			else:
				# Fallback: use first class
				safe_proba = proba_arr[0][1]
			confidence_percent = safe_proba * 100 if safe_proba <= 1 else safe_proba
			prediction_label = 'Safe' if confidence_percent >= 60 else 'Phishing'
			frontend_features = {
				'url_length': feature_dict.get('url_length', 0),
				'num_dots': feature_dict.get('num_dots', 0),
				'has_at_symbol': bool(feature_dict.get('has_at_symbol', 0)),
				'has_ip': bool(feature_dict.get('has_ip', 0)),
				'suspicious_keywords': feature_dict.get('suspicious_keywords', 0)
			}
			return JsonResponse({
				'prediction': prediction_label,
				'confidence': round(confidence_percent, 2),
				'features': frontend_features
			})
		except Exception as e:
			return JsonResponse({'error': str(e)}, status=500)
	else:
		return JsonResponse({'error': 'Only POST allowed'}, status=405)
