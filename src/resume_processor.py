import PyPDF2
import docx
from loguru import logger
from typing import List, Optional, Dict
from src.models import Resume
import re
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import asyncio
import os
from dotenv import load_dotenv
import json
from src.prompts import get_resume_extraction_prompt

# Load environment variables
load_dotenv()


class ResumeProcessor:
    def __init__(self):
        logger.info("Initializing ResumeProcessor")
        self.supported_formats = ['.pdf', '.docx']
        self.client = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            api_key=os.getenv('OPENAI_API_KEY')
        )
        self.embeddings_client = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv('OPENAI_API_KEY')
        )
        self.parser = JsonOutputParser()
        self.prompt_template = get_resume_extraction_prompt()
        logger.debug("Using GPT-4o-mini model for processing")

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            logger.debug(f"Extracting text from PDF: {file_path}")
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            if not text.strip():
                logger.warning(f"No text extracted from PDF: {file_path}")
                return ""

            logger.debug(f"Extracted {len(text)} characters from PDF")
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""

    def _extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            logger.debug(f"Extracting text from DOCX: {file_path}")
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            logger.debug(f"Extracted {len(text)} characters from DOCX")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            return ""

    def _extract_text(self, file_path: str) -> str:
        """Extract text from file based on extension"""
        try:
            logger.debug(f"Extracting text from file: {file_path}")
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension == '.pdf':
                return self._extract_text_from_pdf(file_path)
            elif file_extension == '.docx':
                return self._extract_text_from_docx(file_path)
            else:
                logger.error(f"Unsupported file format: {file_extension}")
                return ""
        except Exception as e:
            logger.error(f"Error extracting text from file {file_path}: {e}")
            return ""

    async def _extract_info_with_openai(self, text: str) -> dict:
        """Extract structured information from resume text using OpenAI"""
        try:
            if not text.strip():
                logger.warning("Empty text provided to OpenAI")
                return {
                    "name": "NA",
                    "email": "NA",
                    "phone": "NA",
                    "location": "NA",
                    "summary": "NA",
                    "skills": {"technical": [], "soft": []},
                    "experience": [],
                    "education": []
                }

            logger.debug("Setting up chain with prompt template")
            chain = self.prompt_template | self.client | self.parser

            logger.debug("Running chain")
            result = await chain.ainvoke({"text": text})

            # Validate and clean the result
            if not isinstance(result, dict):
                logger.warning("Invalid response format from OpenAI")
                return {
                    "name": "NA",
                    "email": "NA",
                    "phone": "NA",
                    "location": "NA",
                    "summary": "NA",
                    "skills": {"technical": [], "soft": []},
                    "experience": [],
                    "education": []
                }

            # Ensure all required fields exist with proper structure
            required_fields = {
                "name": "NA",
                "email": "NA",
                "phone": "NA",
                "location": "NA",
                "summary": "NA",
                "skills": {"technical": [], "soft": []},
                "experience": [],
                "education": []
            }

            for field, default in required_fields.items():
                if field not in result:
                    result[field] = default

            # Clean and validate experience entries
            cleaned_experience = []
            for exp in result.get("experience", []):
                if isinstance(exp, dict):
                    cleaned_exp = {
                        "title": exp.get("title", ""),
                        "company": exp.get("company", ""),
                        "location": exp.get("location", ""),
                        "start_date": exp.get("start_date", ""),
                        "end_date": exp.get("end_date", ""),
                        "currently_working": exp.get("currently_working", False),
                        "description": exp.get("description", "")
                    }
                    cleaned_experience.append(cleaned_exp)
            result["experience"] = cleaned_experience

            # Clean and validate education entries
            cleaned_education = []
            for edu in result.get("education", []):
                if isinstance(edu, dict):
                    cleaned_edu = {
                        "degree": edu.get("degree", ""),
                        "institution": edu.get("institution", ""),
                        "field": edu.get("field", ""),
                        "duration": edu.get("duration", "")
                    }
                    cleaned_education.append(cleaned_edu)
            result["education"] = cleaned_education

            logger.debug("Successfully processed OpenAI response")
            return result

        except Exception as e:
            logger.error(f"Error extracting information with OpenAI: {e}")
            return {
                "name": "NA",
                "email": "NA",
                "phone": "NA",
                "location": "NA",
                "summary": "NA",
                "skills": {"technical": [], "soft": []},
                "experience": [],
                "education": []
            }

    async def process_resume(self, file_path: str) -> Resume:
        """Process a resume file and extract information"""
        try:
            logger.info(f"Processing resume: {file_path}")
            text = self._extract_text(file_path)

            if not text.strip():
                logger.warning(f"No text extracted from resume: {file_path}")
                return Resume(
                    name="NA",
                    email="NA",
                    phone="NA",
                    text="",
                    skills={"technical": [], "soft": []},
                    experience=[],
                    education=[]
                )

            # Extract structured information
            info = await self._extract_info_with_openai(text)

            # Create Resume object
            resume = Resume(
                name=info.get("name", "NA"),
                email=info.get("email", "NA"),
                phone=info.get("phone", "NA"),
                text=text,
                skills=info.get("skills", {"technical": [], "soft": []}),
                experience=info.get("experience", []),
                education=info.get("education", [])
            )

            logger.info(f"Successfully processed resume: {file_path}")
            logger.debug(f"Extracted name: {resume.name}, email: {resume.email}, phone: {resume.phone}")
            logger.debug(
                f"Extracted {len(resume.skills['technical'])} technical skills, {len(resume.skills['soft'])} soft skills, {len(resume.experience)} experiences, and {len(resume.education)} education entries")

            return resume
        except Exception as e:
            logger.error(f"Error processing resume: {e}")
            return Resume(
                name="NA",
                email="NA",
                phone="NA",
                text="",
                skills={"technical": [], "soft": []},
                experience=[],
                education=[]
            )

    def _extract_name(self, text: str) -> str:
        """Extract name from resume text"""
        try:
            logger.debug("Extracting name from resume text")
            # Look for common name patterns
            name_patterns = [
                r"^[A-Z][a-z]+ [A-Z][a-z]+$",  # First Last
                r"^[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+$",  # First M. Last
            ]

            for line in text.split('\n'):
                line = line.strip()
                for pattern in name_patterns:
                    if re.match(pattern, line):
                        logger.debug(f"Found name: {line}")
                        return line
            logger.warning("No name found in resume text")
            return "Unknown"
        except Exception as e:
            logger.error(f"Error extracting name: {e}")
            return "Unknown"

    def _extract_email(self, text: str) -> Optional[str]:
        """Extract email from resume text"""
        try:
            logger.debug("Extracting email from resume text")
            email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            match = re.search(email_pattern, text)
            if match:
                logger.debug(f"Found email: {match.group(0)}")
                return match.group(0)
            logger.warning("No email found in resume text")
            return None
        except Exception as e:
            logger.error(f"Error extracting email: {e}")
            return None

    def _extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number from resume text"""
        try:
            logger.debug("Extracting phone number from resume text")
            phone_patterns = [
                r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',  # US format
                r'\+?\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'  # International format
            ]

            for pattern in phone_patterns:
                match = re.search(pattern, text)
                if match:
                    logger.debug(f"Found phone number: {match.group(0)}")
                    return match.group(0)
            logger.warning("No phone number found in resume text")
            return None
        except Exception as e:
            logger.error(f"Error extracting phone number: {e}")
            return None

    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text"""
        try:
            logger.debug("Extracting skills from resume text")
            # Common skill keywords
            skill_keywords = [
                'python', 'java', 'javascript', 'sql', 'aws', 'azure', 'gcp',
                'machine learning', 'data analysis', 'project management',
                'communication', 'leadership', 'teamwork'
            ]

            skills = []
            for keyword in skill_keywords:
                if keyword.lower() in text.lower():
                    skills.append(keyword)
            logger.debug(f"Found {len(skills)} skills in resume text")
            return skills
        except Exception as e:
            logger.error(f"Error extracting skills: {e}")
            return []

    def _extract_experience(self, text: str) -> List[str]:
        """Extract work experience from resume text"""
        try:
            logger.debug("Extracting experience from resume text")
            # Look for common experience section headers
            experience_headers = ['experience', 'work experience', 'employment history']
            experience = []

            lines = text.split('\n')
            for i, line in enumerate(lines):
                if any(header in line.lower() for header in experience_headers):
                    # Get next few lines as experience
                    for j in range(i + 1, min(i + 10, len(lines))):
                        if lines[j].strip():
                            experience.append(lines[j].strip())
            logger.debug(f"Found {len(experience)} experience entries in resume text")
            return experience
        except Exception as e:
            logger.error(f"Error extracting experience: {e}")
            return []

    def _extract_education(self, text: str) -> List[str]:
        """Extract education from resume text"""
        try:
            logger.debug("Extracting education from resume text")
            # Look for common education section headers
            education_headers = ['education', 'academic background', 'qualifications']
            education = []

            lines = text.split('\n')
            for i, line in enumerate(lines):
                if any(header in line.lower() for header in education_headers):
                    # Get next few lines as education
                    for j in range(i + 1, min(i + 10, len(lines))):
                        if lines[j].strip():
                            education.append(lines[j].strip())
            logger.debug(f"Found {len(education)} education entries in resume text")
            return education
        except Exception as e:
            logger.error(f"Error extracting education: {e}")
            return []

    def _get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for the given text using OpenAIEmbeddings"""
        try:
            logger.debug("Getting embeddings for text")
            if not text.strip():
                logger.warning("Empty text provided for embeddings")
                return []

            # Create embeddings using OpenAIEmbeddings
            embeddings = self.embeddings_client.embed_query(text)

            if not embeddings:
                logger.warning("No embeddings received from OpenAI")
                return []

            logger.debug(f"Successfully generated embeddings of length {len(embeddings)}")
            return embeddings
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return []