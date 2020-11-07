from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from generate_summary import  load_model, generate_summary
import os

base_path = os.getcwd()
dir = os.path.join(base_path,'content/t5/models')
model, tokenizer = load_model(dir)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")




@app.get('/')
def redirect():
    return RedirectResponse("/index")
@app.get('/index')
def return_summary(request: Request):
    result = ''
    return templates.TemplateResponse('index.html', context={'request': request, 'summary': result})
@app.post('/index')
def return_summary(request: Request, input_text: str = Form(...), summary_len: int = Form(150), beams: int = Form(2)):
    summary = generate_summary(input_text, summary_len, beams, model, tokenizer)
    return templates.TemplateResponse('index.html', context={'request': request, 'summary': summary, 'text':input_text})

