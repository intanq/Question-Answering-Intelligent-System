import os
from flask import Flask, flash, render_template, request, redirect
from werkzeug.utils import secure_filename
from image_to_string import recognize_image_to_string
from qa_model import predict_answer, format_answer
from context_retriever import query_to_text, summarize_context


uploads_folder = os.path.join('static', 'uploads')


app = Flask(__name__)
app.secret_key = "secret_key"
app.config['UPLOAD_FOLDER'] = uploads_folder
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'image' not in request.files:  # the name of the input image component is "image"
            flash('No file part')
            print('No file part')
            return redirect(request.url)

        image = request.files['image']
        if image.filename == '':
            flash('No image selected for uploading')
            print('No image selected for uploading')
            return redirect(request.url)

        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(f'upload_image filename: {filename}')
            flash('Image successfully uploaded and displayed below')
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # Image recognition
            question = recognize_image_to_string(full_filename).lower()
            # Get the chosen type of context
            chosen_context = request.form['contexts']
            formatted_answer = ''
            if chosen_context == '1':                 # Web scrapping one website and not summarise it
                contexts_google = query_to_text(question, n=3)
                answer, probability = predict_answer(question, contexts_google[0])
                # formatted_answer = answer, probability
                formatted_answer = format_answer(answer, probability)
            elif chosen_context == '2':               # Web scrapping from one website and summarise it
                summarised_context = summarize_context(query_to_text(question, n=3)[0])
                answer, probability = predict_answer(question, summarised_context)
                formatted_answer = format_answer(answer, probability)
            else:                                     # Provide your own context
                context = request.form['typeYourContext']
                answer, probability = predict_answer(question, context)
                formatted_answer = format_answer(answer, probability)

            print(formatted_answer)
            return render_template('answer.html', question=question.capitalize(), answer=formatted_answer, filename=full_filename)
    return render_template('frontend.html')


if __name__ == '__main__':
    app.run(debug=True)
