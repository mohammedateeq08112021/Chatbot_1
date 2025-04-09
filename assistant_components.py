"""Components for the Interactive AI Assistant."""

import re
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import openai

class URLFeatureExtractor:
    """Extracts features from URLs that can indicate malicious intent."""
    
    def __init__(self):
        # List of suspicious terms often found in malicious URLs
        self.suspicious_terms = [
            'login', 'signin', 'verify', 'secure', 'account', 'update', 'confirm',
            'banking', 'password', 'credential', 'wallet', 'bitcoin', 'payment'
        ]
        
        # List of popular brands often targeted by phishing
        self.target_brands = [
            'paypal', 'apple', 'microsoft', 'amazon', 'netflix', 'google',
            'facebook', 'instagram', 'twitter', 'linkedin', 'bank', 'chase',
            'wellsfargo', 'citibank', 'amex', 'visa', 'mastercard'
        ]
    
    def extract_features(self, url):
        """Extract features from a URL and return a dictionary."""
        features = {}
        
        # Basic URL characteristics
        features['url_length'] = len(url)
        features['has_https'] = 1 if url.startswith('https://') else 0
        features['has_http'] = 1 if url.startswith('http://') else 0
        features['num_dots'] = url.count('.')
        features['num_digits'] = sum(c.isdigit() for c in url)
        features['num_special_chars'] = sum(not c.isalnum() and not c.isspace() for c in url)
        
        # Domain information
        # For simplicity, we'll use a basic domain extraction
        domain_parts = url.split('://')[-1].split('/')[0].split('.')
        features['domain'] = domain_parts[-2] if len(domain_parts) > 1 else ''
        features['subdomain'] = '.'.join(domain_parts[:-2]) if len(domain_parts) > 2 else ''
        features['tld'] = domain_parts[-1] if len(domain_parts) > 0 else ''
        features['domain_length'] = len(features['domain'])
        features['has_subdomain'] = 1 if features['subdomain'] else 0
        
        # Check for suspicious terms
        url_lower = url.lower()
        features['has_suspicious_terms'] = 0
        features['suspicious_terms_found'] = []
        for term in self.suspicious_terms:
            if term in url_lower:
                features['has_suspicious_terms'] = 1
                features['suspicious_terms_found'].append(term)
        
        # Check for brand terms (potential phishing)
        features['has_brand_terms'] = 0
        features['brand_terms_found'] = []
        for brand in self.target_brands:
            if brand in url_lower:
                features['has_brand_terms'] = 1
                features['brand_terms_found'].append(brand)
        
        # Path analysis
        path = url.split('://')[-1].split('/', 1)[-1] if '/' in url.split('://')[-1] else ''
        features['path_length'] = len(path)
        features['num_path_segments'] = path.count('/')
        
        # Query parameters
        features['has_query_params'] = 1 if '?' in url else 0
        features['num_query_params'] = url.count('&') + (1 if '?' in url else 0)
        
        # Check for obfuscation techniques
        features['has_obfuscation'] = 0
        if '%' in url or '@' in url or 'data:' in url or 'javascript:' in url:
            features['has_obfuscation'] = 1
        
        return features

class URLReputationChecker:
    """Checks URLs against reputation services."""
    
    def __init__(self):
        # In a real implementation, you would initialize API clients here
        pass
    
    def check_url_reputation(self, url):
        """Check URL reputation and return results."""
        results = {
            'safe_browsing': self._check_safe_browsing(url),
            'phishtank': self._check_phishtank(url),
            'domain_age': self._check_domain_age(url),
            'ssl_info': self._check_ssl(url)
        }
        return results
    
    def _check_safe_browsing(self, url):
        """Simulate checking against Google Safe Browsing."""
        # In a real implementation, this would call the Safe Browsing API
        # For simulation, we'll use some heuristics
        if 'malware' in url or 'virus' in url or 'hack' in url:
            return {'is_safe': False, 'threats': ['malware']}
        return {'is_safe': True, 'threats': []}
    
    def _check_phishtank(self, url):
        """Simulate checking against PhishTank."""
        # In a real implementation, this would call the PhishTank API
        # For simulation, we'll use some heuristics
        suspicious_patterns = ['secure', 'login', 'signin', 'account', 'update', 'verify']
        if any(pattern in url.lower() for pattern in suspicious_patterns):
            return {'is_phishing': True, 'confidence': 0.9}
        return {'is_phishing': False, 'confidence': 0.1}
    
    def _check_domain_age(self, url):
        """Check domain registration information."""
        try:
            # Extract domain
            domain_parts = url.split('://')[-1].split('/')[0].split('.')
            domain = '.'.join(domain_parts[-2:]) if len(domain_parts) > 1 else domain_parts[0]
            
            # In a real implementation, this would use the whois library
            # For simulation, we'll return a random age
            import random
            days_old = random.randint(1, 3650)  # 1 day to 10 years
            
            return {
                'domain': domain,
                'days_old': days_old,
                'is_new': days_old < 30  # Consider domains less than 30 days old as new
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _check_ssl(self, url):
        """Check SSL certificate information."""
        if url.startswith('https://'):
            # In a real implementation, this would check the SSL certificate
            # For simulation, we'll return a random validity
            import random
            is_valid = random.choice([True, True, True, False])  # 75% chance of valid
            
            return {
                'has_ssl': True,
                'is_valid': is_valid,
                'issuer': 'Simulated CA' if is_valid else 'Unknown',
                'expiry_days': random.randint(1, 365) if is_valid else 0
            }
        else:
            return {'has_ssl': False}

class URLAnalyzer:
    """Main URL analyzer that integrates feature extraction and reputation checking."""
    
    def __init__(self):
        self.feature_extractor = URLFeatureExtractor()
        self.reputation_checker = URLReputationChecker()
    
    def analyze_url(self, url):
        """Analyze a URL and return comprehensive results."""
        # Extract features
        features = self.feature_extractor.extract_features(url)
        
        # Check reputation
        reputation = self.reputation_checker.check_url_reputation(url)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(features, reputation)
        
        # Determine risk level
        risk_level = self._determine_risk_level(risk_score)
        
        # Compile analysis results
        analysis = {
            'url': url,
            'features': features,
            'reputation': reputation,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'timestamp': datetime.now().isoformat()
        }
        
        return analysis
    
    def _calculate_risk_score(self, features, reputation):
        """Calculate a risk score based on features and reputation."""
        score = 0
        
        # Feature-based scoring
        if features['url_length'] > 100:
            score += 10
        if not features['has_https']:
            score += 20
        if features['num_dots'] > 3:
            score += 10
        if features['has_suspicious_terms']:
            score += 15 * len(features['suspicious_terms_found'])
        if features['has_brand_terms']:
            score += 15 * len(features['brand_terms_found'])
        if features['has_obfuscation']:
            score += 25
        
        # Reputation-based scoring
        if not reputation['safe_browsing']['is_safe']:
            score += 50
        if reputation['phishtank']['is_phishing']:
            score += 40 * reputation['phishtank']['confidence']
        if reputation['domain_age'].get('is_new', False):
            score += 25
        if not reputation['ssl_info'].get('has_ssl', False):
            score += 15
        elif not reputation['ssl_info'].get('is_valid', True):
            score += 30
        
        return min(score, 100)  # Cap at 100
    
    def _determine_risk_level(self, risk_score):
        """Determine risk level based on risk score."""
        if risk_score < 20:
            return 'low'
        elif risk_score < 50:
            return 'medium'
        elif risk_score < 80:
            return 'high'
        else:
            return 'critical'
    
    def get_user_recommendation(self, analysis):
        """Generate user-friendly recommendations based on analysis."""
        risk_level = analysis['risk_level']
        features = analysis['features']
        reputation = analysis['reputation']
        
        # Determine action based on risk level
        if risk_level == 'low':
            action = 'This URL appears to be safe.'
        elif risk_level == 'medium':
            action = 'Exercise caution when visiting this site.'
        elif risk_level == 'high':
            action = 'Avoid visiting this site unless absolutely necessary.'
        else:  # critical
            action = 'Do not visit this site under any circumstances.'
        
        # Generate explanation
        explanation = f"This URL has been analyzed and determined to have a {risk_level.upper()} risk level."
        
        # Compile details/reasons
        details = []
        
        # Add feature-based details
        if not features['has_https']:
            details.append("The site does not use HTTPS encryption.")
        if features['has_suspicious_terms']:
            terms = ', '.join(features['suspicious_terms_found'])
            details.append(f"The URL contains suspicious terms: {terms}.")
        if features['has_brand_terms']:
            brands = ', '.join(features['brand_terms_found'])
            details.append(f"The URL contains brand terms that might indicate phishing: {brands}.")
        if features['num_dots'] > 3:
            details.append("The URL contains an unusual number of dots.")
        if features['has_obfuscation']:
            details.append("The URL contains obfuscation techniques.")
        
        # Add reputation-based details
        if not reputation['safe_browsing']['is_safe']:
            threats = ', '.join(reputation['safe_browsing']['threats'])
            details.append(f"The URL has been flagged for: {threats}.")
        if reputation['phishtank']['is_phishing']:
            details.append("The URL has characteristics of a phishing site.")
        if reputation['domain_age'].get('is_new', False):
            days = reputation['domain_age'].get('days_old', 0)
            details.append(f"The domain is relatively new (approximately {days} days old).")
        if not reputation['ssl_info'].get('has_ssl', False):
            details.append("The site does not use SSL encryption.")
        elif not reputation['ssl_info'].get('is_valid', True):
            details.append("The site's SSL certificate is invalid or self-signed.")
        
        # Add safety tips based on risk level
        safety_tips = []
        if risk_level in ['medium', 'high', 'critical']:
            safety_tips = [
                "Never enter personal or financial information on suspicious websites.",
                "Look for HTTPS and a padlock icon in your browser before entering sensitive information.",
                "Check for spelling errors and unusual domains in URLs.",
                "Be cautious of websites asking for unnecessary information.",
                "When in doubt, navigate to websites directly rather than following links."
            ]
        
        # Compile recommendation
        recommendation = {
            'risk_level': risk_level,
            'action': action,
            'explanation': explanation,
            'details': details,
            'safety_tips': safety_tips
        }
        
        return recommendation

class ThreatDetectionModel:
    """Detects if a URL is a security threat based on its analysis."""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
    
    def detect_url_threat(self, url_analysis):
        """Detect if a URL is a threat based on its analysis."""
        # For simplicity, we'll use a rule-based approach
        # In a real implementation, this would use a trained ML model
        result = self._rule_based_detection(url_analysis)
        
        return result
    
    def _rule_based_detection(self, url_analysis):
        """Rule-based threat detection."""
        risk_score = url_analysis['risk_score']
        features = url_analysis['features']
        reputation = url_analysis['reputation']
        
        # Determine if it's a threat
        is_threat = False
        threat_types = []
        confidence = 0.0
        
        # Check for phishing
        phishing_indicators = [
            features['has_brand_terms'],
            features['has_suspicious_terms'],
            reputation['phishtank']['is_phishing']
        ]
        phishing_score = sum(phishing_indicators) / len(phishing_indicators)
        
        # Check for malware
        malware_indicators = [
            not reputation['safe_browsing']['is_safe'],
            not reputation['ssl_info'].get('is_valid', True) if reputation['ssl_info'].get('has_ssl', False) else False,
            features['has_obfuscation']
        ]
        malware_score = sum(1 for indicator in malware_indicators if indicator) / len(malware_indicators)
        
        # Determine threat type and confidence
        if phishing_score > 0.5 or malware_score > 0.5 or risk_score >= 50:
            is_threat = True
            
            if phishing_score > 0.5:
                threat_types.append('phishing')
            
            if malware_score > 0.5:
                threat_types.append('malware')
            
            # If no specific threat type identified but risk is high
            if not threat_types and risk_score >= 50:
                threat_types.append('suspicious')
            
            # Calculate overall confidence
            confidence_scores = []
            if phishing_score > 0.5:
                confidence_scores.append(phishing_score)
            if malware_score > 0.5:
                confidence_scores.append(malware_score)
            if risk_score >= 50:
                confidence_scores.append(risk_score / 100)
            
            confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        return {
            'is_threat': is_threat,
            'threat_types': threat_types,
            'confidence': confidence
        }

class AIAssistant:
    """Handles educational queries using AI models."""
    
    def __init__(self, api_key=None, model="gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.client = None
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
    
    def set_api_key(self, api_key):
        """Set the OpenAI API key."""
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
    
    def set_model(self, model):
        """Set the model to use."""
        self.model = model
    
    def get_response(self, query, conversation_history=None):
        """Get a response from the AI model."""
        if not self.client:
            return "Error: API key not set. Please set an API key to use the AI assistant."
        
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": "You are a helpful educational assistant specializing in cybersecurity. Provide clear, accurate, and educational responses to user queries."}
            ]
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history)
            
            # Add the current query
            messages.append({"role": "user", "content": query})
            
            # Get response from OpenAI using the new API format
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_streaming_response(self, query, conversation_history=None):
        """Get a streaming response from the AI model."""
        if not self.client:
            yield "Error: API key not set. Please set an API key to use the AI assistant."
            return
        
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": "You are a helpful educational assistant specializing in cybersecurity. Provide clear, accurate, and educational responses to user queries."}
            ]
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history)
            
            # Add the current query
            messages.append({"role": "user", "content": query})
            
            # Get streaming response from OpenAI using the new API format
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
            )
            
            # Process the streaming response
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Error: {str(e)}"

class QueryRouter:
    """Routes queries to the appropriate component based on content."""
    
    def __init__(self):
        # URL pattern for basic detection
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
    
    def classify_query(self, query):
        """Classify a query as URL analysis or educational."""
        if self.url_pattern.match(query):
            return "URL_ANALYSIS", query
        else:
            # Check if the query contains a URL
            urls = self.url_pattern.findall(query)
            if urls:
                return "URL_ANALYSIS", urls[0]
            else:
                return "EDUCATIONAL_QUERY", query

class InteractiveAssistant:
    """Main application class that integrates all components."""
    
    def __init__(self, openai_api_key=None):
        self.url_analyzer = URLAnalyzer()
        self.threat_detector = ThreatDetectionModel()
        self.ai_assistant = AIAssistant(api_key=openai_api_key)
        self.query_router = QueryRouter()
        self.conversation_history = []
    
    def set_api_key(self, api_key):
        """Set the OpenAI API key."""
        self.ai_assistant.set_api_key(api_key)
    
    def set_model(self, model):
        """Set the AI model to use."""
        self.ai_assistant.set_model(model)
    
    def process_query(self, query):
        """Process a user query and return the appropriate response."""
        # Classify the query
        query_type, processed_query = self.query_router.classify_query(query)
        
        # Add user query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Process based on query type
        if query_type == "URL_ANALYSIS":
            # Analyze the URL
            analysis = self.url_analyzer.analyze_url(processed_query)
            threat_detection = self.threat_detector.detect_url_threat(analysis)
            recommendation = self.url_analyzer.get_user_recommendation(analysis)
            
            # Format the response
            response = self._format_url_analysis_response(processed_query, analysis, threat_detection, recommendation)
        else:
            # Get response from AI assistant
            ai_response = self.ai_assistant.get_response(processed_query, self.conversation_history[:-1])
            response = ai_response
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _format_url_analysis_response(self, url, analysis, threat_detection, recommendation):
        """Format URL analysis results into a user-friendly response."""
        response = f"URL ANALYSIS RESULT:\n\n"
        response += f"URL: {url}\n"
        response += f"Risk Level: {recommendation['risk_level'].upper()}\n\n"
        
        response += f"Action: {recommendation['action']}\n\n"
        
        if recommendation['details']:
            response += "Reasons:\n"
            for detail in recommendation['details']:
                response += f"- {detail}\n"
            response += "\n"
        
        if threat_detection['is_threat']:
            threat_types = ", ".join(threat_detection['threat_types'])
            response += f"Threat Type: {threat_types}\n"
            response += f"Confidence: {threat_detection['confidence']:.2f}\n\n"
        
        if recommendation.get('safety_tips'):
            response += "Safety Tips:\n"
            for tip in recommendation['safety_tips']:
                response += f"- {tip}\n"
            response += "\n"
        
        response += "This analysis is based on automated checks and heuristics. Always exercise caution when visiting unfamiliar websites."
        
        return response
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        return "Conversation history cleared."
