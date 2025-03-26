from openai import OpenAI
import argparse

parser = argparse.ArgumentParser(description='train for SLG')
parser.add_argument('--dataset', default="phoenix2014", help='the target dataset')
args = parser.parse_args()

target = args.dataset # phoenix2014, phoenix2014T, CSLDaily

api_key="xxxx"

model_name = 'gpt-4o'  
temperature = 0.7

client = OpenAI(api_key=api_key)

with open('prompt_{}.txt'.format(target), 'r', encoding='utf-8') as pt:
    body = pt.readline()

with open('words_{}.txt'.format(target), 'r', encoding='utf-8') as input_file:
    words = input_file.readlines()
    for idx, word in enumerate(words):
        prompt = body + word

        completion = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    
        sens = completion.choices[0].message.content

        with open('description_{}.txt'.format(target), 'a', encoding='utf-8') as output_file:
            output_file.write(sens)
            output_file.write('\n')
            output_file.write('\n')

        print("generating gloss" + str(idx+1))
