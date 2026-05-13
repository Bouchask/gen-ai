import ollama
import json
import logging
import re
import ast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_json_string(s):
    """Attempt to fix common LLM JSON errors."""
    # Remove everything before the first '[' and after the last ']'
    match = re.search(r'\[.*\]', s, re.DOTALL)
    if match:
        s = match.group(0)
    
    # Remove markdown code blocks if they exist
    s = s.replace("```json", "").replace("```", "").strip()
    
    # Fix trailing commas before closing brackets
    s = re.sub(r',\s*\]', ']', s)
    s = re.sub(r',\s*\}', '}', s)
    
    # Try to handle unescaped internal double quotes in "text": "..." 
    # This is a common failure point for LLMs.
    def escape_internal_quotes(match):
        prefix = match.group(1) # ' "text": "'
        content = match.group(2) # the actual text content
        suffix = match.group(3) # '" }' or '", '
        # Escape double quotes only within the content
        content = content.replace('"', '\\"')
        return f'{prefix}{content}{suffix}'
    
    # Match the pattern of the text value
    s = re.sub(r'("text":\s*")(.*?)("\s*[,}\]])', escape_internal_quotes, s, flags=re.DOTALL)
    
    return s

def generate_podcast_script(topic: str, model: str = "llama3"):
    """
    Generates a deep-dive dialogue between 'Host' and 'Expert'.
    Returns a list of dictionaries: [{"speaker": "Host", "text": "..."}]
    """
    prompt = f"""
    Create a professional 4-minute podcast script about: {topic}.
    
    Characters:
    - Host: Enthusiastic, inquisitive, asks technical follow-ups.
    - Expert: Specialist, uses specific industrial examples, provides detailed explanations.
    
    Requirements:
    - Exactly 60 back-and-forth dialogue lines.
    - Expert responses should be 2-3 sentences long.
    - Format: STRICT JSON list of objects with "speaker" and "text" keys.
    
    CRITICAL CONSTRAINTS:
    - DO NOT use any double quotes (") inside the 'text' values. Use single quotes (') for titles or emphasis instead.
    - NO newlines inside the 'text' values.
    - Return ONLY the JSON list. No preamble. No sign-off.
    
    Example:
    [
      {{"speaker": "Host", "text": "Welcome! We are talking about 'Smart Factories' today."}},
      {{"speaker": "Expert", "text": "Exactly. AI is the backbone of modern automation."}}
    ]
    """
    
    logging.info(f"Generating detailed 4-minute script for topic: {topic}")
    
    try:
        response = ollama.chat(
            model=model, 
            messages=[{'role': 'system', 'content': "You are a professional script writer who only outputs valid JSON."},
                      {'role': 'user', 'content': prompt}],
            options={'num_predict': 8192, 'temperature': 0.2}, # Low temperature for consistency
            keep_alive=0 
        )
        
        content = response['message']['content']
        cleaned_content = clean_json_string(content)
        
        try:
            script = json.loads(cleaned_content)
        except Exception:
            logging.warning("JSON standard parse failed, trying literal_eval...")
            # Fallback for minor syntax errors
            script = ast.literal_eval(cleaned_content)
            
        logging.info(f"Successfully generated script with {len(script)} lines.")
        return script
    except Exception as e:
        logging.error(f"Error generating script: {e}")
        return []

if __name__ == "__main__":
    test_script = generate_podcast_script("The future of AI in Smart Factories")
    print(json.dumps(test_script, indent=2))
