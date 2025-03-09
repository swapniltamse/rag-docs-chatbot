import os
import glob
import PyPDF2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Directory containing your documents
docs_dir = "static_docs"

print(f"Looking for documents in {os.path.abspath(docs_dir)}")

# Check if directory exists
if not os.path.exists(docs_dir):
    print(f"ERROR: Directory {docs_dir} does not exist!")
    os.makedirs(docs_dir)
    print(f"Created directory {docs_dir}")
else:
    print(f"Directory {docs_dir} exists")

# List all files
all_files = list(glob.glob(f"{docs_dir}/**/*.*", recursive=True))
print(f"Found {len(all_files)} files in total")
for file in all_files:
    print(f" - {file}")

# Process PDF files
pdf_files = list(glob.glob(f"{docs_dir}/**/*.pdf", recursive=True))
print(f"\nFound {len(pdf_files)} PDF files")

for pdf_path in pdf_files:
    print(f"\nProcessing {pdf_path}")
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            print(f" - PDF has {len(pdf_reader.pages)} pages")
            
            # Try to extract text from the first page as a test
            if len(pdf_reader.pages) > 0:
                text = pdf_reader.pages[0].extract_text()
                if text and text.strip():
                    print(f" - Successfully extracted text from first page")
                    print(f" - First 200 chars: {text[:200].replace(chr(10), ' ')}")
                else:
                    print(f" - WARNING: First page has no extractable text!")
                    print(f" - This PDF might be scanned images or have security restrictions")
    except Exception as e:
        print(f" - ERROR processing PDF: {str(e)}")