from transformers import BertTokenizer
import torch.nn as nn

tokenizer = BertTokenizer.from_pretrained('kykim/bert-kor-base')

#토크나이저
def Bert_Tokenizer(text_list, maximum_length = 16) :
    attention_masks = []
    input_ids = []

    for text in text_list :
        encoded = tokenizer.encode_plus(text, 
                                add_special_tokens=True ,
                                max_length=maximum_length,
                                truncation=True,
                                padding='max_length',
                                return_tensors='pt'
                                )
        attention_masks.append(encoded['attention_mask'])
        input_ids.append(encoded['input_ids'])

    return input_ids, attention_masks

class BertClassifier(nn.Module) :
    def __init__(self, bert_model, hidden_size, classes) :
        super(BertClassifier, self).__init__()

        self.bert = bert_model
        self.hidden_size = hidden_size
        self.classes = classes

        self.fc1 = nn.Linear(self.bert.config.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(self.hidden_size, self.classes)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask) :
        output = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        cls = output.pooler_output
        
        x = self.fc1(cls)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x