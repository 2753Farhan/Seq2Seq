#!/bin/bash

ROLL_NUMBER="YOUR_BSSE_ROLL"  # Change this to your actual roll number

echo "Preparing submission for $ROLL_NUMBER..."

# Create PDF of report (requires pandoc)
if command -v pandoc &> /dev/null; then
    pandoc REPORT.md -o ${ROLL_NUMBER}.pdf
    echo "PDF created: ${ROLL_NUMBER}.pdf"
else
    echo "Warning: pandoc not installed. Please convert REPORT.md to PDF manually."
fi

# Create GitHub repository URL file
echo "https://github.com/YOUR_USERNAME/seq2seq-code-generation" > ${ROLL_NUMBER}_github.txt

# Package everything
zip -r ${ROLL_NUMBER}_submission.zip \
    ${ROLL_NUMBER}.pdf \
    ${ROLL_NUMBER}_github.txt \
    README.md \
    REPORT.md \
    src/ \
    config/ \
    scripts/ \
    docker/ \
    requirements.txt \
    setup.py

echo "Submission package created: ${ROLL_NUMBER}_submission.zip"