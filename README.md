# Resume-Matching-with-Job-Descriptions-Using-PDF-CVs
Building a PDF extractor to pull relevant details from CVs in PDF format, and matching them against the job descriptions from the Hugging Face dataset.
Approach to the Task:
The task involved matching job descriptions with candidate resumes based on their content similarity. Here's a summary of the approach used:
Extraction:
- Wrote a PDF text extraction tool that extracts the text from a given file
- Wrote a Python script that created a resume.csv file that has 3 columns i.e. 'I.D' which should be the name of the pdf file, 'resume_content' which should be the extracted text from the pdf, and 'category' which should be the folder in which the pdf file is present.

 Data Preprocessing:
   - Loaded resume and job description data from CSV files.
   - Handled missing values in the data by removing rows with missing content.
   - Ensured uniform formatting of the 'category' column in the resume data.
   - Performed text preprocessing on resume contents, including tokenization, removal of stopwords, and stemming.

Job Description Encoding:
   - Utilized a pre-trained NLP model (DistilBERT) to encode job descriptions into fixed-size embeddings.

Candidate-Job Matching:
   - Embedded resume contents using the same NLP model.
   - Calculated cosine similarity between each resume and all job descriptions.
   - Selected the top 5 matching resumes for each job description based on similarity scores.

Results Export:
   - Stored the matching results, including company names, position titles, resume IDs, and similarity scores, in a CSV file ('matching_results.csv').
