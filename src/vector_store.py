import pinecone
from pinecone import Pinecone, ServerlessSpec
from loguru import logger
import os
from typing import List, Dict, Any
from src.models import Resume, Job
import numpy as np


class VectorStore:
    def __init__(self):
        logger.info("Initializing Pinecone vector store")
        # Initialize Pinecone with new API
        self.pc = Pinecone(
            api_key=os.getenv('PINECONE_API_KEY')
        )

        # Create or connect to index
        self.index_name = "resume-matcher"
        if self.index_name not in self.pc.list_indexes().names():
            logger.info(f"Creating new Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',  # Using gcp-starter for free tier compatibility
                    region='us-east-1'
                )
            )

        self.index = self.pc.Index(self.index_name)
        logger.info("Successfully connected to Pinecone index")

    def store_resume(self, resume: Resume, embedding: List[float]) -> None:
        """Store resume embedding in Pinecone"""
        try:
            logger.debug(f"Storing resume: {resume.name}")
            # Convert any None values to empty strings or lists for metadata
            # Convert education and experience to string lists
            education_list = [str(edu) for edu in (resume.education or [])]
            experience_list = [str(exp) for exp in (resume.experience or [])]

            # Convert skills dictionary to flat list of strings
            skills_list = []
            if resume.skills:
                if isinstance(resume.skills, dict):
                    # Handle dictionary format with soft and technical skills
                    soft_skills = resume.skills.get('soft', [])
                    tech_skills = resume.skills.get('technical', [])
                    skills_list = [str(skill) for skill in soft_skills + tech_skills]
                else:
                    # Handle list format
                    skills_list = [str(skill) for skill in resume.skills]

            metadata = {
                "type": "resume",
                "name": resume.name or "",
                "email": resume.email or "",
                "phone": resume.phone or "",
                "skills": skills_list,
                "experience": experience_list,
                "education": education_list
            }

            # Generate a unique ID for the resume
            resume_id = f"resume_{hash(resume.name)}"

            # Upsert the embedding
            self.index.upsert(
                vectors=[(resume_id, embedding, metadata)],
                namespace="resumes"
            )
            logger.info(f"Successfully stored resume: {resume.name}")
        except Exception as e:
            logger.error(f"Error storing resume in Pinecone: {e}")
            raise

    def store_job(self, job: Job, embedding: List[float]) -> None:
        """Store job embedding in Pinecone"""
        try:
            logger.debug(f"Storing job: {job.title} at {job.company}")
            # Convert any None values to empty strings for metadata
            # Convert requirements to string list
            requirements_list = [str(req) for req in (job.requirements or [])]

            metadata = {
                "type": "job",
                "title": job.title or "",
                "company": job.company or "",
                "location": job.location or "",
                "salary": job.salary or "",
                "requirements": requirements_list
            }

            # Generate a unique ID for the job
            job_id = f"job_{hash(f'{job.title}_{job.company}')}"

            # Upsert the embedding
            self.index.upsert(
                vectors=[(job_id, embedding, metadata)],
                namespace="jobs"
            )
            logger.info(f"Successfully stored job: {job.title} at {job.company}")
        except Exception as e:
            logger.error(f"Error storing job in Pinecone: {e}")
            raise

    def find_matches(self, resume: Resume, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find matching jobs for a resume using vector similarity"""
        try:
            logger.debug(f"Finding matches for resume: {resume.name}")
            # Get resume embedding
            resume_embedding = self._get_resume_embedding(resume)

            # Query Pinecone for similar jobs
            results = self.index.query(
                vector=resume_embedding,
                top_k=top_k,
                namespace="jobs",
                include_metadata=True
            )

            matches = []
            for match in results.matches:
                metadata = match.metadata
                score = match.score
                matches.append({
                    'job_title': metadata['title'],
                    'company': metadata['company'],
                    'location': metadata['location'],
                    'score': score,
                    'requirements': metadata['requirements']
                })

            logger.info(f"Found {len(matches)} matches for resume: {resume.name}")
            return matches
        except Exception as e:
            logger.error(f"Error finding matches in Pinecone: {e}")
            return []

    def _get_resume_embedding(self, resume: Resume) -> List[float]:
        """Get embedding for resume text"""
        try:
            # Combine relevant text for embedding
            text_parts = []

            # Add main text
            if resume.text:
                text_parts.append(resume.text)

            # Add skills
            if resume.skills:
                if isinstance(resume.skills, dict):
                    # Handle dictionary format with soft and technical skills
                    soft_skills = resume.skills.get('soft', [])
                    tech_skills = resume.skills.get('technical', [])
                    text_parts.extend([str(skill) for skill in soft_skills + tech_skills])
                else:
                    # Handle list format
                    text_parts.extend([str(skill) for skill in resume.skills])

            # Add experience
            if resume.experience:
                text_parts.extend([str(exp) for exp in resume.experience])

            # Combine all text parts
            combined_text = ' '.join(text_parts)

            # Get embedding from OpenAI
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=combined_text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting resume embedding: {e}")
            raise

    def clear_namespace(self, namespace: str) -> None:
        """Clear all vectors from a namespace"""
        try:
            logger.info(f"Clearing namespace: {namespace}")
            # Check if namespace exists before trying to clear it
            try:
                self.index.delete(delete_all=True, namespace=namespace)
                logger.info(f"Successfully cleared namespace: {namespace}")
            except Exception as e:
                if "Namespace not found" in str(e):
                    logger.warning(f"Namespace {namespace} does not exist, skipping clear")
                else:
                    raise
        except Exception as e:
            logger.error(f"Error clearing namespace: {e}")
            raise
