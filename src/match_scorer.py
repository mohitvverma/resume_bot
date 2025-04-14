from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from loguru import logger
import os
from dotenv import load_dotenv
from typing import Dict, List, Tuple

# Load environment variables
load_dotenv()

MATCH_SCORING_PROMPT = """
You are an expert resume-job matching specialist. Analyze the following job and resume details to provide a match score and detailed justification.

Job Details:
{job_details}

Resume Details:
{resume_details}

Provide a comprehensive analysis including:
1. Match Score (1-10)
2. Detailed Justification
3. Key Matching Points
4. Areas for Improvement

Return the analysis in the following JSON format:
{{
    "match_score": int,  // Score between 1-10
    "justification": str,  // Detailed explanation of the score
    "key_matching_points": [str],  // List of key matching points
    "areas_for_improvement": [str]  // List of areas where the candidate could improve
}}

Be precise and objective in your analysis. Do not make assumptions beyond the provided information.
"""


class MatchScorer:
    def __init__(self):
        logger.info("Initializing MatchScorer")
        self.client = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            api_key=os.getenv('OPENAI_API_KEY')
        )
        self.parser = JsonOutputParser()
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert resume-job matching specialist. Provide objective analysis."),
            ("user", MATCH_SCORING_PROMPT)
        ])

    async def score_match(self, job_details: Dict, resume_details: Dict) -> Dict:
        """Score the match between a job and resume"""
        try:
            logger.debug("Scoring job-resume match")

            # Format details for the prompt
            formatted_job = self._format_job_details(job_details)
            formatted_resume = self._format_resume_details(resume_details)

            # Create chain
            chain = self.prompt_template | self.client | self.parser

            # Get analysis
            result = await chain.ainvoke({
                "job_details": formatted_job,
                "resume_details": formatted_resume
            })

            # Validate result
            if not isinstance(result, dict):
                logger.warning("Invalid response format from OpenAI")
                return self._get_default_response()

            # Ensure score is between 1-10
            if "match_score" in result:
                result["match_score"] = max(1, min(10, int(result["match_score"])))

            logger.debug(f"Match score: {result.get('match_score', 'N/A')}")
            return result

        except Exception as e:
            logger.error(f"Error scoring match: {e}")
            return self._get_default_response()

    def _format_job_details(self, job: Dict) -> str:
        """Format job details for the prompt"""
        return f"""
Title: {job.get('title', 'N/A')}
Company: {job.get('company', 'N/A')}
Description: {job.get('description', 'N/A')}
Requirements: {job.get('requirements', 'N/A')}
Location: {job.get('location', 'N/A')}
Salary: {job.get('salary', 'N/A')}
"""

    def _format_resume_details(self, resume: Dict) -> str:
        """Format resume details for the prompt"""
        return f"""
Name: {resume.get('name', 'N/A')}
Email: {resume.get('email', 'N/A')}
Phone: {resume.get('phone', 'N/A')}
Location: {resume.get('location', 'N/A')}
Summary: {resume.get('summary', 'N/A')}

Technical Skills: {', '.join(resume.get('skills', {}).get('technical', []))}
Soft Skills: {', '.join(resume.get('skills', {}).get('soft', []))}

Experience:
{self._format_experience(resume.get('experience', []))}

Education:
{self._format_education(resume.get('education', []))}
"""

    def _format_experience(self, experience: List[Dict]) -> str:
        """Format experience details"""
        formatted = []
        for exp in experience:
            formatted.append(f"""
- Title: {exp.get('title', 'N/A')}
  Company: {exp.get('company', 'N/A')}
  Location: {exp.get('location', 'N/A')}
  Duration: {exp.get('start_date', 'N/A')} - {exp.get('end_date', 'N/A')}
  Currently Working: {exp.get('currently_working', False)}
  Description: {exp.get('description', 'N/A')}
""")
        return "\n".join(formatted)

    def _format_education(self, education: List[Dict]) -> str:
        """Format education details"""
        formatted = []
        for edu in education:
            formatted.append(f"""
- Degree: {edu.get('degree', 'N/A')}
  Institution: {edu.get('institution', 'N/A')}
  Field: {edu.get('field', 'N/A')}
  Duration: {edu.get('duration', 'N/A')}
""")
        return "\n".join(formatted)

    def _get_default_response(self) -> Dict:
        """Return default response in case of error"""
        return {
            "match_score": 1,
            "justification": "Unable to analyze match due to processing error",
            "key_matching_points": [],
            "areas_for_improvement": []
        }