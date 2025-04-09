# from langgraph.graph import StateGraph, END, MessagesState 
# from langgraph.prebuilt import ToolNode 
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_community.tools import TavilySearchResults 
# from langchain.agents import load_tools

def process(model, tokenizer):
    prompt = 'Вот текст, в котором нужно произвести замену. Заменить нужно словосочетание "очень красивый" ' \
            'Я стоял на балконе и смотрел на этот очень красивый закат над морем'
    sys_prompt = '''
        Ты - ассистент по подбору синонимов
                                    
        Твоя задача состоит в замене одиночных прилагательных
        или прилагательных, которым предшествует слово "очень" на более
        подходящие синонимы.
                                    
        Ты должен опираться на контекст и стиль, в которых употреблено прилагательное, и предлагать
        несколько вариантов подходящих замен. 
                                    
        Если тебе не хватает данных, используй инструмент TavilySearchResults 
        для поиска синонимов к заданному слову или словосочетанию.
                                    
        Ответь списком из пяти наиболее подходящих слов, в порядке убывания твоей уверенности в них.
        Никаких других данных в ответе быть не должно, только слова-синонимы.
        '''

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response
