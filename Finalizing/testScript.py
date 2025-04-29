#!/usr/bin/env python3
import sys
import requests
from lxml import html
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def get_best_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

# Wikipedia fetch

def search_and_fetch_article(topic):
    search_url = 'https://en.wikipedia.org/w/api.php'
    search_params = {
        'action': 'query', 'format': 'json',
        'list': 'search', 'utf8': 1, 'srsearch': topic
    }
    resp = requests.get(search_url, params=search_params, timeout=5).json()
    hits = resp.get('query', {}).get('search', [])
    if not hits:
        return ""
    title = hits[0]['title']
    parse_params = {'action': 'parse', 'format': 'json', 'page': title, 'prop': 'text', 'redirects': ''}
    page = requests.get(search_url, params=parse_params, timeout=5).json()
    raw = page['parse']['text']['*']
    doc = html.fromstring(raw)
    paras = doc.xpath('//p')
    return "\n\n".join([p.text_content().strip() for p in paras if p.text_content().strip()])

# Main

def main():
    device = get_best_device()
    print(f"Using device: {device}")

    # load models
    q_tok = T5Tokenizer.from_pretrained('./results')
    q_model = T5ForConditionalGeneration.from_pretrained('./results').to(device)
    a_tok = q_tok
    a_model = T5ForConditionalGeneration.from_pretrained('./results/answer_results').to(device)

    topics = []
    for i in range(5):
        t = input(f"Enter topic #{i+1}: ").strip()
        topics.append(t)

    difficulties = [100, 200, 300, 400, 500]

    for topic in topics:
        print(f"\nCategory: {topic}")
        art = search_and_fetch_article(topic)
        if not art:
            print("  [No article found]")
            continue
        # generate 5 questions
        inputs = q_tok(art, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        input_ids = inputs.input_ids.to(device)
        outputs = q_model.generate(
            input_ids,
            num_beams=5,
            num_return_sequences=5,
            max_length=64,
            early_stopping=True
        )
        questions = [q_tok.decode(o, skip_special_tokens=True) for o in outputs]

        for diff, q in zip(difficulties, questions):
            print(f"  ${diff}: Q: {q}")
            # generate answer
            ai = a_tok(q, return_tensors='pt', truncation=True, padding='max_length', max_length=64)
            aid = ai.input_ids.to(device)
            ao = a_model.generate(aid, max_length=64)
            ans = a_tok.decode(ao[0], skip_special_tokens=True)
            print(f"       A: {ans}")

if __name__ == '__main__':
    main()
