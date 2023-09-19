import os
import pdfplumber
import pandas as pd

# Define the path to the directory containing resume categories (subdirectories)
path_to_resume_categories = os.getcwd() + '/resumes'

# List the categories (subdirectories) within the specified directory
list_of_categories = os.listdir(path_to_resume_categories)

# Create an empty DataFrame to store the extracted resume data
df = pd.DataFrame(columns=['I.D', 'resume_text', 'Category'])

# Iterate through each category (subdirectory) of resumes
for each_category in list_of_categories:
    # List the PDF files within each category subdirectory
    list_of_pdf_in_each_category = os.listdir(path_to_resume_categories + '/' + each_category)
    
    # Process up to the first 5 PDF files in each category
    for each_pdf in list_of_pdf_in_each_category[:5]:
        # Construct the full path to the PDF file
        pdf_path = path_to_resume_categories + '/' + each_category + '/' + each_pdf
        
        # Open the PDF file using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            # Extract text from each page of the PDF and concatenate it
            for page in pdf.pages:
                text += page.extract_text()
        
        # Create a row of data for the DataFrame with the PDF's I.D, text, and category
        row_data = [[each_pdf, text, each_category]]
        row_df = pd.DataFrame(row_data, columns=['I.D', 'resume_text', 'Category'])
        
        # Merge the current row DataFrame with the main DataFrame using an outer join
        df = pd.merge(df, row_df, how='outer')

# Save the final DataFrame as a CSV file without the index
df.to_csv('resume.csv', index=False)
