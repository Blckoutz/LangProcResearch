from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the pre-trained T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Example text for summarization
input_text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.
Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
Colloquially, the term "artificial intelligence" is often used to describe machines (or computers) that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem-solving".
As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect.
"""

# Prepare the input with the summarization prefix
input_text_with_prefix = "summarize: " + input_text

# Tokenize the input text
input_ids = tokenizer.encode(input_text_with_prefix, return_tensors="pt", max_length=512, truncation=True)

# Generate summary with the model
summary_ids = model.generate(input_ids, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

# Decode the summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Output the summary
print("Original Text: ")
print(input_text)
print("\nGenerated Summary: ")
print(summary)
