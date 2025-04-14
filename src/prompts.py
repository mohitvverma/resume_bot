from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

RESUME_EXTRACTION_PROMPT = """
You are a professional resume parser. Analyze the following resume text and extract the following information in a structured JSON format. Do not make any assumptions beyond what is explicitly stated in the text. If information is not found, use "NA" as the value.

Required Information:
1. Personal Information:
   - name: Full name of the candidate
   - email: Email address
   - phone: Phone number
   - location: Current location (city, state/country)

2. Professional Summary:
   - summary: A concise overview of the candidate's professional background and key skills

3. Work Experience:
   - title: Job title
   - company: Company name
   - location: Job location
   - start_date: Start date (MM/YYYY)
   - end_date: End date (MM/YYYY) or "Present"
   - currently_working: Boolean indicating if this is the current role
   - description: Key responsibilities and achievements

4. Education:
   - degree: Degree obtained
   - institution: Educational institution
   - field: Field of study
   - duration: Duration of study

5. Skills:
   - technical: List of technical skills
   - soft: List of soft skills

Return the information in the following JSON format:
{{
    "name": "string",
    "email": "string",
    "phone": "string",
    "location": "string",
    "summary": "string",
    "experience": [
        {{
            "title": "string",
            "company": "string",
            "location": "string",
            "start_date": "string",
            "end_date": "string",
            "currently_working": boolean,
            "description": "string"
        }}
    ],
    "education": [
        {{
            "degree": "string",
            "institution": "string",
            "field": "string",
            "duration": "string"
        }}
    ],
    "skills": {{
        "technical": ["string"],
        "soft": ["string"]
    }}
}}

Resume Text:
{text}
"""


def get_resume_extraction_prompt():
    """
    Creates and returns a ChatPromptTemplate for resume extraction.
    Includes proper error handling and logging.

    Returns:
        ChatPromptTemplate: A chat prompt template initialized with a JSON output parser.
    """
    try:
        logger.info("Initializing resume extraction prompt with JSON output parser")

        template = ChatPromptTemplate.from_messages([
            ("system", "You are a professional resume parser. Extract information exactly as specified."),
            ("user", RESUME_EXTRACTION_PROMPT)
        ])

        return template
    except Exception as e:
        logger.exception(f"Error creating resume extraction prompt: {e}")
        raise 