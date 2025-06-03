#!/usr/bin/env python3
"""
WeirdBench: A script to evaluate LLM response weirdness
"""

import json
import pdb
import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
import anthropic
import google.generativeai as genai
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

JUDGE_PROMPT_INDIVIDUAL = """You are a judge evaluating how weird or unusual a set of LLM responses is. 
Rate the weirdness of the responses on a scale from 1 to 100, where:
- 1-20: Completely normal, expected responses with good variety
- 21-40: Slightly unusual but still reasonable
- 41-60: Moderately weird, with some unexpected elements or patterns
- 61-80: Very weird, with significant repetition or strange patterns
- 81-100: Extremely weird, with heavy repetition or bizarre patterns across all responses

Consider both individual the following factors:
- Relevance to the question
- Logical consistency
- Unusual word choices or phrasing
- Unexpected tangents or topics
- Overall coherence

Respond with ONLY a number between 1 and 100. Do not include any other text."""

JUDGE_PROMPT_AGGREGATE = """You are a judge evaluating how weird or unusual a set of LLM responses is. 
Rate the weirdness of the responses on a scale from 1 to 100, where:
- 1-20: Completely normal, expected responses with good variety
- 21-40: Slightly unusual but still reasonable
- 41-60: Moderately weird, with some unexpected elements or patterns
- 61-80: Very weird, with significant repetition or strange patterns
- 81-100: Extremely weird, with heavy repetition or bizarre patterns across all responses

When evaluating weirdness, focus on the overall pattern of the responses. The following things are weird and should be scored highly:
- Repetition of phrases or ideas across different responses
- Similar sentence structures or patterns that appear multiple times
- Unusual word choices that appear repeatedly
- Unexpected tangents that are repeated across responses
- Overall variety vs. repetition

Respond with ONLY a number between 1 and 100. Do not include any other text."""




def retry_with_backoff(func, max_retries=3, base_delay=1, max_delay=60):
    """Retry function with exponential backoff for API errors."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if it's a retryable error
            retryable_errors = [
                'overloaded', 'rate limit', 'timeout', 'service unavailable',
                '429', '503', '529', 'too many requests'
            ]
            
            is_retryable = any(err in error_msg for err in retryable_errors)
            
            if not is_retryable or attempt == max_retries - 1:
                print(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print(f"Max retries reached. Skipping this request.")
                    return None
                raise e
            
            delay = min(base_delay * (2 ** attempt), max_delay)
            print(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
    
    return None

def load_questions(data_file: str = "data.json") -> List[str]:
    """Load all questions from the data.json file."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    questions = []
    for category, category_questions in data["sample_llm_questions"].items():
        questions.extend(category_questions)
    
    return questions

def load_config(config_dir: str) -> Dict[str, Any]:
    """Load configuration from the specified directory."""
    config_path = Path(config_dir) / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    required_fields = ["system_prompt", "model_name"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    
    return config

def query_llm(openai_client: OpenAI, anthropic_client: anthropic.Anthropic, 
              model: str, system_prompt: str, question: str) -> str:
    """Query the LLM with a single question, supporting multiple providers."""
    
    # OpenAI models (including xAI via OpenAI-compatible API)
    if model.startswith(("gpt-", "grok-", 'o3')):
        def make_request():
            if model.startswith("grok-"):
                # Use xAI client for Grok models
                xai_client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
                return xai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
            else:
                # Use OpenAI client for GPT models
                return openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
        
        response = retry_with_backoff(make_request)
        if response is None:
            return f"Error: Failed to get response from {model} after retries"
        return response.choices[0].message.content
    
    # Anthropic Claude models
    elif model.startswith("claude-"):
        def make_request():
            return anthropic_client.messages.create(
                model=model,
                max_tokens=1000,
                temperature=0.7,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": question}
                ]
            )
        
        response = retry_with_backoff(make_request)
        if response is None:
            return f"Error: Failed to get response from {model} after retries"
        return response.content[0].text
    
    # Google Gemini models
    elif model.startswith("gemini-"):
        def make_request():
            genai_model = genai.GenerativeModel(model, system_instruction=system_prompt)
            return genai_model.generate_content(
                question,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1000
                )
            )
        
        response = retry_with_backoff(make_request)
        if response is None:
            return f"Error: Failed to get response from {model} after retries"
        return response.text
    
    else:
        raise ValueError(f"Unsupported model: {model}")

def query_judge_gpt4o_mini(openai_client: OpenAI, questions: List[str], responses: List[str], combined: bool = False) -> int:
    """Query GPT-4o-mini to rate the weirdness of responses."""
    if combined:
        # Combine all questions and responses
        combined_input = "Questions and Responses:\n\n"
        for q, r in zip(questions, responses):
            combined_input += f"Question: {q}\nResponse: {r}\n\n"
        combined_input += "Weirdness score (1-100):"
        
        def make_request():
            return openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT_AGGREGATE},
                    {"role": "user", "content": combined_input}
                ],
                temperature=0.1,
                max_tokens=10
            )
    else:
        # Evaluate each response individually
        scores = []
        for q, r in zip(questions, responses):
            individual_input = f"Question: {q}\nResponse: {r}\n\nWeirdness score (1-100):"
            
            def make_request():
                return openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": JUDGE_PROMPT_INDIVIDUAL},
                        {"role": "user", "content": individual_input}
                    ],
                    temperature=0.1,
                    max_tokens=10
                )
            
            response = retry_with_backoff(make_request)
            if response is None:
                print("Failed to get GPT-4o-mini judge response, using default score")
                scores.append(-1)
            else:
                score = parse_score(response.choices[0].message.content.strip(), "GPT-4o-mini")
                scores.append(score)
            time.sleep(0.3)
        
        # Return average of individual scores
        valid_scores = [s for s in scores if s != -1]
        return sum(valid_scores) / len(valid_scores) if valid_scores else -1
    
    judge_response = retry_with_backoff(make_request)
    if judge_response is None:
        print("Failed to get GPT-4o-mini judge response, using default score")
        return -1
    
    return parse_score(judge_response.choices[0].message.content.strip(), "GPT-4o-mini")

def query_judge_claude_haiku(anthropic_client: anthropic.Anthropic, questions: List[str], responses: List[str], combined: bool = False) -> int:
    """Query Claude 3.5 Haiku to rate the weirdness of responses."""
    if combined:
        # Combine all questions and responses
        combined_input = "Questions and Responses:\n\n"
        for q, r in zip(questions, responses):
            combined_input += f"Question: {q}\nResponse: {r}\n\n"
        combined_input += "Weirdness score (1-100):"
        
        def make_request():
            return anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=10,
                temperature=0.1,
                system=JUDGE_PROMPT_AGGREGATE,
                messages=[
                    {"role": "user", "content": combined_input}
                ]
            )
    else:
        # Evaluate each response individually
        scores = []
        for q, r in zip(questions, responses):
            individual_input = f"Question: {q}\nResponse: {r}\n\nWeirdness score (1-100):"
            
            def make_request():
                return anthropic_client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=10,
                    temperature=0.1,
                    system=JUDGE_PROMPT_INDIVIDUAL,
                    messages=[
                        {"role": "user", "content": individual_input}
                    ]
                )
            
            response = retry_with_backoff(make_request)
            if response is None:
                print("Failed to get Claude 3.5 Haiku judge response, using default score")
                scores.append(-1)
            else:
                score = parse_score(response.content[0].text.strip(), "Claude 3.5 Haiku")
                scores.append(score)
            time.sleep(0.3)
        
        # Return average of individual scores
        valid_scores = [s for s in scores if s != -1]
        return sum(valid_scores) / len(valid_scores) if valid_scores else -1
    
    judge_response = retry_with_backoff(make_request)
    if judge_response is None:
        print("Failed to get Claude 3.5 Haiku judge response, using default score")
        return -1
    
    return parse_score(judge_response.content[0].text.strip(), "Claude 3.5 Haiku")

def query_judge_gemini_flash(questions: List[str], responses: List[str], combined: bool = False) -> int:
    """Query Gemini Flash to rate the weirdness of responses."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    if combined:
        # Combine all questions and responses
        combined_input = "Questions and Responses:\n\n"
        for q, r in zip(questions, responses):
            combined_input += f"Question: {q}\nResponse: {r}\n\n"
        combined_input += "Weirdness score (1-100):"
        
        # Combine system prompt and user input for Gemini
        full_prompt = f"{JUDGE_PROMPT_AGGREGATE}\n\n{combined_input}"
        
        def make_request():
            return model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=10
                )
            )
    else:
        # Evaluate each response individually
        scores = []
        for q, r in zip(questions, responses):
            individual_input = f"Question: {q}\nResponse: {r}\n\nWeirdness score (1-100):"
            full_prompt = f"{JUDGE_PROMPT_INDIVIDUAL}\n\n{individual_input}"
            
            def make_request():
                return model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=10
                    )
                )
            
            response = retry_with_backoff(make_request)
            if response is None:
                print("Failed to get Gemini Flash judge response, using default score")
                scores.append(-1)
            else:
                score = parse_score(response.text.strip(), "Gemini Flash")
                scores.append(score)
            time.sleep(0.3)
        
        # Return average of individual scores
        valid_scores = [s for s in scores if s != -1]
        return sum(valid_scores) / len(valid_scores) if valid_scores else -1
    
    judge_response = retry_with_backoff(make_request)
    if judge_response is None:
        print("Failed to get Gemini Flash judge response, using default score")
        return -1
    
    return parse_score(judge_response.text.strip(), "Gemini Flash")

def query_judge_grok_3(xai_client: OpenAI, questions: List[str], responses: List[str], combined: bool = False) -> int:
    """Query Grok 3 to rate the weirdness of responses."""
    if combined:
        # Combine all questions and responses
        combined_input = "Questions and Responses:\n\n"
        for q, r in zip(questions, responses):
            combined_input += f"Question: {q}\nResponse: {r}\n\n"
        combined_input += "Weirdness score (1-100):"
        
        def make_request():
            return xai_client.chat.completions.create(
                model="grok-3",
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT_AGGREGATE},
                    {"role": "user", "content": combined_input}
                ],
                temperature=0.1,
                max_tokens=10
            )
    else:
        # Evaluate each response individually
        scores = []
        for q, r in zip(questions, responses):
            individual_input = f"Question: {q}\nResponse: {r}\n\nWeirdness score (1-100):"
            
            def make_request():
                return xai_client.chat.completions.create(
                    model="grok-3",
                    messages=[
                        {"role": "system", "content": JUDGE_PROMPT_INDIVIDUAL},
                        {"role": "user", "content": individual_input}
                    ],
                    temperature=0.1,
                    max_tokens=10
                )
            
            response = retry_with_backoff(make_request)
            if response is None:
                print("Failed to get Grok-3 judge response, using default score")
                scores.append(-1)
            else:
                score = parse_score(response.choices[0].message.content.strip(), "Grok 3")
                scores.append(score)
            time.sleep(0.3)
        
        # Return average of individual scores
        valid_scores = [s for s in scores if s != -1]
        return sum(valid_scores) / len(valid_scores) if valid_scores else -1
    
    judge_response = retry_with_backoff(make_request)
    if judge_response is None:
        print("Failed to get Grok-3 judge response, using default score")
        return -1
    
    return parse_score(judge_response.choices[0].message.content.strip(), "Grok 3")

def parse_score(score_text: str, judge_name: str) -> int:
    """Parse the numeric score from judge response."""
    import re
    numbers = re.findall(r'\d+', score_text)
    if numbers:
        score = int(numbers[0])
        return max(1, min(100, score))  # Clamp between 1 and 100
    else:
        print(f"Warning: Could not parse {judge_name} score from: {score_text}")
        return -1  # Default middle score

def query_all_judges(openai_client: OpenAI, anthropic_client: anthropic.Anthropic, 
                    xai_client: OpenAI, questions: List[str], responses: List[str], combined: bool = False) -> List[Dict]:
    """Query all judge models to evaluate the responses."""
    judges = []
    
    # GPT-4o-mini
    score = query_judge_gpt4o_mini(openai_client, questions, responses, combined)
    judges.append({"name": "GPT-4o-mini", "score": score})
    time.sleep(0.3)
    
    # Claude 3.5 Haiku
    score = query_judge_claude_haiku(anthropic_client, questions, responses, combined)
    judges.append({"name": "Claude 3.5 Haiku", "score": score})
    time.sleep(0.3)
    
    # Gemini Flash
    score = query_judge_gemini_flash(questions, responses, combined)
    judges.append({"name": "Gemini Flash", "score": score})
    time.sleep(0.3)
    
    # Grok 3
    score = query_judge_grok_3(xai_client, questions, responses, combined)
    judges.append({"name": "Grok 3", "score": score})
    time.sleep(0.3)
    
    return judges

def save_responses(responses: List[Dict], output_dir: str):
    """Save all responses to a JSON file."""
    output_path = Path(output_dir) / "responses.json"
    with open(output_path, 'w') as f:
        json.dump(responses, f, indent=2)

def save_scores(scores_data: Dict, output_dir: str):
    """Save detailed scoring data to JSON and summary to score.txt."""
    # Save detailed scores to JSON
    scores_path = Path(output_dir) / "detailed_scores.json"
    with open(scores_path, 'w') as f:
        json.dump(scores_data, f, indent=2)
    
    # Calculate statistics
    individual_scores = []
    pattern_scores = []
    
    for judge in scores_data["individual_judge_scores"]:
        if judge["score"] != -1:
            individual_scores.append(judge["score"])
    
    for judge in scores_data["pattern_judge_scores"]:
        if judge["score"] != -1:
            pattern_scores.append(judge["score"])
    
    individual_avg = sum(individual_scores) / len(individual_scores) if individual_scores else -1
    pattern_avg = sum(pattern_scores) / len(pattern_scores) if pattern_scores else -1
    
    # Save summary to score.txt
    score_path = Path(output_dir) / "score.txt"
    with open(score_path, 'w') as f:
        f.write("WeirdBench Results\n")
        f.write("==================\n\n")
        
        f.write("All Questions and Responses:\n")
        f.write("-" * 50 + "\n")
        
        for i, (question, response) in enumerate(zip(scores_data["questions"], scores_data["responses"]), 1):
            f.write(f"Question {i}: {question[:50]}...\n")
            f.write(f"Response: {response[:200]}...\n\n")
        
        f.write(f"Overall Statistics:\n")
        f.write(f"Total Questions: {len(scores_data['questions'])}\n")
        f.write(f"Judge Models Used: GPT-4o-mini, Claude 3.5 Haiku, Gemini Flash, Grok 3\n\n")
        
        f.write("Individual Response Scores:\n")
        f.write("-" * 20 + "\n")
        for judge in scores_data["individual_judge_scores"]:
            score_str = f"{judge['score']}" if judge["score"] != -1 else "N/A"
            f.write(f"{judge['name']}: {score_str}\n")
        avg_str = f"{individual_avg:.2f}" if individual_avg != -1 else "N/A"
        f.write(f"Average: {avg_str}\n\n")
        
        f.write("Pattern Analysis Scores:\n")
        f.write("-" * 20 + "\n")
        for judge in scores_data["pattern_judge_scores"]:
            score_str = f"{judge['score']}" if judge["score"] != -1 else "N/A"
            f.write(f"{judge['name']}: {score_str}\n")
        avg_str = f"{pattern_avg:.2f}" if pattern_avg != -1 else "N/A"
        f.write(f"Average: {avg_str}\n")

def process_question(question: str, config: Dict[str, Any], openai_client: OpenAI, anthropic_client: anthropic.Anthropic) -> Dict:
    """Process a single question and get LLM response."""
    response = query_llm(openai_client, anthropic_client, config["model_name"], config["system_prompt"], question)
    return {
        "question": question,
        "response": response,
        "model": config["model_name"],
        "system_prompt": config["system_prompt"]
    }

def process_judges(responses: List[Dict], openai_client: OpenAI, anthropic_client: anthropic.Anthropic, xai_client: OpenAI) -> Dict:
    """Process all responses with judges."""
    questions = [r["question"] for r in responses]
    response_texts = [r["response"] for r in responses]
    
    # Get individual scores
    individual_judge_scores = query_all_judges(
        openai_client, anthropic_client, xai_client,
        questions, response_texts, combined=False
    )
    
    # Get combined pattern scores
    pattern_judge_scores = query_all_judges(
        openai_client, anthropic_client, xai_client,
        questions, response_texts, combined=True
    )
    
    # Calculate averages
    individual_valid_scores = [judge["score"] for judge in individual_judge_scores if judge["score"] != -1]
    individual_avg = sum(individual_valid_scores) / len(individual_valid_scores) if individual_valid_scores else -1
    
    pattern_valid_scores = [judge["score"] for judge in pattern_judge_scores if judge["score"] != -1]
    pattern_avg = sum(pattern_valid_scores) / len(pattern_valid_scores) if pattern_valid_scores else -1
    
    return {
        "questions": questions,
        "responses": response_texts,
        "individual_judge_scores": individual_judge_scores,
        "pattern_judge_scores": pattern_judge_scores,
        "individual_average_score": individual_avg,
        "pattern_average_score": pattern_avg
    }

def main():
    parser = argparse.ArgumentParser(description="Run WeirdBench evaluation with 4 judge models")
    parser.add_argument("--config_dir", '-c', help="Directory containing config.json")
    parser.add_argument("--data-file", '-d', default="data.json", help="Path to questions data file")
    parser.add_argument("--max-workers", '-w', type=int, default=8, help="Maximum number of parallel workers")
    args = parser.parse_args()

    # Setup API clients
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    xai_api_key = os.getenv("XAI_API_KEY")
    
    if not openai_api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    if not anthropic_api_key:
        print("Error: Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)
        
    if not google_api_key:
        print("Error: Google API key required. Set GOOGLE_API_KEY environment variable.")
        sys.exit(1)
        
    if not xai_api_key:
        print("Error: xAI API key required. Set XAI_API_KEY environment variable.")
        sys.exit(1)
    
    openai_client = OpenAI(api_key=openai_api_key)
    anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
    genai.configure(api_key=google_api_key)
    xai_client = OpenAI(api_key=xai_api_key, base_url="https://api.x.ai/v1")
    
    # Load configuration and questions
    print(f"Loading config from: {args.config_dir}")
    config = load_config(args.config_dir)
    
    print(f"Loading questions from: {args.data_file}")
    questions = load_questions(args.data_file)
    
    print(f"Loaded {len(questions)} questions")
    print(f"Model: {config['model_name']}")
    print(f"Using 4 judge models: GPT-4o-mini, Claude 3.5 Haiku, Gemini Flash, Grok 3")
    print(f"Using {args.max_workers} parallel workers")
    
    # Create output directory
    output_dir = Path(args.config_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Query LLM for all questions in parallel
    print("\nQuerying LLM in parallel...")
    responses = []
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_question = {
            executor.submit(process_question, question, config, openai_client, anthropic_client): question 
            for question in questions
        }
        
        for i, future in enumerate(as_completed(future_to_question), 1):
            question = future_to_question[future]
            try:
                response_data = future.result()
                responses.append(response_data)
                print(f"Completed question {i}/{len(questions)}: {question[:50]}...")
            except Exception as e:
                print(f"Error processing question {i}: {str(e)}")
    
    # Save responses
    save_responses(responses, output_dir)
    print(f"\nSaved responses to: {output_dir}/responses.json")
    
    # Get weirdness scores from judges for all responses together
    print("\nEvaluating all responses together with judge LLMs...")
    
    scores_data = process_judges(responses, openai_client, anthropic_client, xai_client)
    
    # Save scores
    save_scores(scores_data, output_dir)
    print(f"\nSaved detailed scores to: {output_dir}/detailed_scores.json")
    print(f"Saved score summary to: {output_dir}/score.txt")
    
    # Print final results
    individual_valid_scores = [judge["score"] for judge in scores_data["individual_judge_scores"] if judge["score"] != -1]
    pattern_valid_scores = [judge["score"] for judge in scores_data["pattern_judge_scores"] if judge["score"] != -1]
    
    individual_avg = sum(individual_valid_scores) / len(individual_valid_scores) if individual_valid_scores else -1
    pattern_avg = sum(pattern_valid_scores) / len(pattern_valid_scores) if pattern_valid_scores else -1
    
    print("\nFinal Scores:")
    print(f"Individual Response Average: {individual_avg:.2f}" if individual_avg != -1 else "N/A")
    print(f"Pattern Analysis Average: {pattern_avg:.2f}" if pattern_avg != -1 else "N/A")

if __name__ == "__main__":
    main() 