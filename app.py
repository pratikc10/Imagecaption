from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer ,ViltProcessor, ViltForQuestionAnswering
import torch
from PIL import Image
from flask import Flask, request, jsonify,render_template
# from PIL import Image
import os
import openai
from google.cloud import vision
import io

app = Flask(__name__)


#openapi settings
openai.api_type = "azure"
openai.api_base = "https://dattaraj-openai-demo.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = "56dc6d2fdf8c48debea0493b8db17bfa"
deployment_id="deployment-5358c80585d74d4d987ca7f18f674bb8"

#gcp api settings
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.curdir, r'C:\Users\pratik_chauhan1\projects\ImageCaption\mlconsole-poc-9b7d0b77642a.json')
# print(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
# client = vision.ImageAnnotatorClient()

# model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# max_length = 16
# num_beams = 4
# gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

#textImage Model
processor1 = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model1 = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def textImage(image_paths,text):
    image1 = image_paths
    text = text
    image = Image.open(image1)
    # prepare inputs
    encoding = processor1(image, text, return_tensors="pt")

# forward pass
    outputs = model1(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    image_answer=  model1.config.id2label[idx]

    return image_answer

# def predict_step(image_paths):
#   images = []
#   for image_path in image_paths:
#     i_image = Image.open(image_path)
#     if i_image.mode != "RGB":
#       i_image = i_image.convert(mode="RGB")

#     images.append(i_image)

#   pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
#   pixel_values = pixel_values.to(device)

#   output_ids = model.generate(pixel_values, **gen_kwargs)

#   preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#   preds = [pred.strip() for pred in preds]
#   return preds

def vision_api(con):
    content = con
    client = vision.ImageAnnotatorClient()
    image1 = vision.Image(content=content)
    response = client.label_detection(image=image1)
    labels = response.label_annotations
    vision1 =[label.description for label in labels]
    print(vision1)
    # for label in labels:
    #     print(label.description)
    return vision1
def vison_text(con):
    content = con
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    # print("Texts:")
    data =[]
    for text in texts:
        data.append(text.description)
      
    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    return data
@app.route('/sendmessage', methods=['POST'])
def sendmessage():
    print(request.files)
    image = request.files['image']
    print(image)
    con=image.read()
    if request.form['message'] :
        text = request.form['message']
    else :
        text = "Image description in detail"
    # if text : 
    # ans1= "Question -"+text+" and The ans of question  "+textImage(image,text)+" . The image contains these objects {}".format()
    # print("this ans",ans1)
    image_caption =textImage(image,text)
    print(image_caption)
    vision_ans = vision_api(con)
    visiontext=vison_text(con)
    # # ans2 = predict_step([image])
    # # print("this is ans1 by textImage model-- ",ans1)
    # # answer1= ans1[-1]
    # # answer2 = ans2[-1]
    # answer=[]
    # answer.append(ans1)
    # answer.append(vision_ans)
    # ans = str(answer)
    if visiontext:
        # print("Texts:",visiontext[0])
        ans1= """
        modeloutput - The Question:-{} and The  answer of question  is {} . and 
        objects-  The image contains these objects {} and 
        Text - These are the text present in the image {}""".format(text,image_caption,vision_ans,visiontext)
    else:
        ans1= "Question -"+text+" and The answer is "+image_caption+" . and The image contains these objects {} .".format(vision_ans)
    # print(ans1)
    # user_query= "I gave an image and I asked this question to a two mutlimodal image model:  "+text+" . First output by Textimage model gave as output : "+answer1+" and Second Output by  model2 "+answer2+" . \n"+" Using  the response from the  Textimage and model2 , Answer the question in detail by Textimage if found in Textimage then answer according to Textimage and dont go for model2 and dont mention model2 in the  response and   if not found then try to answer by model2 and dont mention Textimage if answer found in model2 : "+text
    # user_query= "I gave an image and I asked this question to a  mutlimodal image models:  "+text+" . First output by mutlimodel  gave as output : "+ans1+" and  "+answer2+" . \n"+" Using  the response from   both , Answer the question  by using both  response and  answer the question according to answer in detail  : "+text
    # user_query= "I gave an image and I asked this question to a mutlimodal image model:  "+text+" . It gave as output : "+ans1+" and "+ans1+" . \n"+" Using  the response from the mulitmodal model , Answer the question in detail : "+text
    # user_query= "I gave an image and I asked this question to a mutlimodel :  "+text+" . It gave as  output  : "+ans1+". Using  the  answers from the mulitmodal model ,give the answer for question in detail and mention the text  id text present in the answer   :"+text+" " 
    # user_query= "I gave an image and I asked this question to a mutlimodel :  "+text+" . It is giving correct  output   : "+ans1+". Using  the  answers1 ,answer2 ,ansnswer3 and from the mulitmodal model ,give the  answer for question {}  according to answer1 and if not found then look for answer2 and answer3 and dont mention answer 1 ,answer 2 and answer 3  word .".format(text)

    user_query= "I gave an image and I asked this question to a mutlimodel :  "+text+" . It is giving correct  output   : "+ans1+". Using  the  answers1 ,objects and text   from the mulitmodal model ,give the   answer using modeloutput if not found then try in objects and if question ask about objects dont consider modeloutput answer and if question ask about text in the image then give answer using text and donot mention where you find answer in multimodal output and give the specific answer to  according question:- {}  ".format(text)

    # else :
    #     ans2 = predict_step([image])
    #     print("this is ans2 by Image model-- ",ans2)
    #     answer = ans2[-1]
    #     text = "Image description"
    #     user_query= "I gave an image and I asked this question to a mutlimodal image model:  "+text+" . It gave as output : "+answer+" . \n"+" Using  the response from the mulitmodal model , Answer the question in detail : "+text
  

    
    # user_query= "I gave an image and I asked this question to a mutlimodal image model:  "+text+" . It gave as output : "+answer+" . \n"+" Using  the response from the mulitmodal model , Answer the question in detail : "+text
    # user_query= "I gave an image and I asked this question to a two mutlimodal image model:  "+text+" . First out by Textimage model gave as output : "+ans1+" and Second Output by image model2 "+answer2+" . \n"+" Using  the response from the  mulitmodal image  Textimage and model2 , Answer the question in detail by Textimage if yes in Textimage then answer according to Textimage and dont go for model2 and dont mention model2 in the  response and   if not found then try to answer by model2 : "+text
    PROMPT = "<|im_start|> system \n The system is an expert on answering questions from given query , The query also includes response obtained from a model that was given input an image  \n<|im_end|>\n<|im_start|>user\n  "+ user_query + "\n<|im_end|>\n<|im_start|>assistant"
    try:
        response1 = openai.Completion.create(
            engine="gpt-35-turbo",
            prompt=PROMPT,
            temperature=1,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["<|im_end|>"])

        final= response1['choices'][0]['text']
        response = final
        print(type(response))
    except :
        response = ["Something Went Wrong , Please Try after sometime"] 
    
    # print("this is response",response)
    return jsonify(response)
    # else :
    # #    ans = predict_step([image])
    # #    answer= ans[-1]
    #    response = answer2
    #    return jsonify(response)

@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

