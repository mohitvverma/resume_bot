import pinecone
from pinecone import Pinecone, ServerlessSpec
from loguru import logger
import os
from typing import List, Dict, Any, Optional
from src.models import Resume, Job
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class VectorStore:
    def __init__(self):
        logger.info("Initializing Pinecone vector store")
        # Initialize Pinecone with new API
        self.pc = Pinecone(
            api_key=os.getenv('PINECONE_API_KEY')
        )

        # Use a single index with different namespaces
        self.index_name = "resume-matcher"
        if self.index_name not in self.pc.list_indexes().names():
            logger.info(f"Creating new Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )

        self.index = self.pc.Index(self.index_name)
        logger.info("Successfully connected to Pinecone index")

        # Define namespaces for different tabs
        self.namespaces = {
            "resume_matching": "resume_matching",  # For Tab 2
            "candidate_search": "candidate_search"  # For Tab 3
        }

    def store_resume(self, resume: Resume, embedding: List[float]) -> None:
        """Store resume embedding in Pinecone (for Tab 2)"""
        try:
            logger.debug(f"Storing resume: {resume.name}")
            metadata = self._prepare_resume_metadata(resume)
            resume_id = f"resume_{hash(resume.name)}"

            # Store in resume_matching namespace
            self.index.upsert(
                vectors=[(resume_id, embedding, metadata)],
                namespace=self.namespaces["resume_matching"]
            )
            logger.info(f"Successfully stored resume: {resume.name}")
        except Exception as e:
            logger.error(f"Error storing resume in Pinecone: {e}")
            raise

    def store_job(self, job: Job, embedding: List[float]) -> None:
        """Store job embedding in Pinecone (for Tab 2)"""
        try:
            logger.debug(f"Storing job: {job.title} at {job.company}")
            metadata = self._prepare_job_metadata(job)
            job_id = f"job_{hash(f'{job.title}_{job.company}')}"

            # Store in resume_matching namespace
            self.index.upsert(
                vectors=[(job_id, embedding, metadata)],
                namespace=self.namespaces["resume_matching"]
            )
            logger.info(f"Successfully stored job: {job.title} at {job.company}")
        except Exception as e:
            logger.error(f"Error storing job in Pinecone: {e}")
            raise

    def store_candidate(self, candidate: Dict, embedding: List[float]) -> None:
        """Store candidate embedding in Pinecone (for Tab 3)"""
        try:
            logger.debug(f"Storing candidate: {candidate.get('name', 'Unknown')}")

            # Prepare metadata
            metadata = {
                'name': candidate.get('name', ''),
                'email': candidate.get('email', ''),
                'phone': candidate.get('phone', ''),
                'current_title': candidate.get('current_title', ''),
                'current_company': candidate.get('current_company', ''),
                'location': candidate.get('location', ''),
                'linkedin': candidate.get('linkedin', ''),
                'skills': candidate.get('skills', []),
                'experience': candidate.get('experience', []),
                'education': candidate.get('education', [])
            }

            # Generate candidate ID
            candidate_id = f"candidate_{hash(candidate.get('email', 'unknown'))}"

            # Store in candidate_search namespace
            self.index.upsert(
                vectors=[(candidate_id, embedding, metadata)],
                namespace=self.namespaces["candidate_search"]
            )
            logger.info(f"Successfully stored candidate: {candidate.get('name', 'Unknown')}")
        except Exception as e:
            logger.error(f"Error storing candidate in Pinecone: {e}")
            raise

    def find_matches(self, resume: Resume, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find matching jobs for a resume using vector similarity (for Tab 2)"""
        try:
            logger.debug(f"Finding matches for resume: {resume.name}")
            resume_embedding = self._get_resume_embedding(resume)

            # Query Pinecone for similar jobs in resume_matching namespace
            results = self.index.query(
                vector=resume_embedding,
                top_k=top_k,
                namespace=self.namespaces["resume_matching"],
                include_metadata=True
            )

            return self._process_match_results(results)
        except Exception as e:
            logger.error(f"Error finding matches in Pinecone: {e}")
            return []

    def find_candidates(self, job: Job, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find matching candidates for a job using vector similarity (for Tab 3)"""
        try:
            # logger.debug(f"Finding candidates for job: {job.title}")
            job_embedding = self._get_job_embedding(job)

            # Query Pinecone for similar candidates in candidate_search namespace
            results = self.index.query(
                vector=job_embedding,
                top_k=top_k,
                namespace=self.namespaces["candidate_search"],
                include_metadata=True
            )

            return self._process_match_results(results)
        except Exception as e:
            logger.error(f"Error finding candidates in Pinecone: {e}")
            return []

    def _prepare_resume_metadata(self, resume: Resume) -> Dict[str, Any]:
        """Prepare metadata for resume storage"""
        education_list = [str(edu) for edu in (resume.education or [])]
        experience_list = [str(exp) for exp in (resume.experience or [])]

        skills_list = []
        if resume.skills:
            if isinstance(resume.skills, dict):
                soft_skills = resume.skills.get('soft', [])
                tech_skills = resume.skills.get('technical', [])
                skills_list = [str(skill) for skill in soft_skills + tech_skills]
            else:
                skills_list = [str(skill) for skill in resume.skills]

        return {
            "type": "resume",
            "name": resume.name or "",
            "email": resume.email or "",
            "phone": resume.phone or "",
            "skills": skills_list,
            "experience": experience_list,
            "education": education_list
        }

    def _prepare_job_metadata(self, job: Job) -> Dict[str, Any]:
        """Prepare metadata for job storage"""
        requirements_list = [str(req) for req in (job.requirements or [])]

        return {
            "type": "job",
            "title": job.title or "",
            "company": job.company or "",
            "location": job.location or "",
            "salary": job.salary or "",
            "requirements": requirements_list
        }

    def _prepare_candidate_metadata(self, candidate: Dict) -> Dict[str, Any]:
        """Prepare metadata for candidate storage"""
        return {
            "type": "candidate",
            "name": candidate.get('name', ''),
            "email": candidate.get('email', ''),
            "phone": candidate.get('phone', ''),
            "current_title": candidate.get('current_title', ''),
            "current_company": candidate.get('current_company', ''),
            "location": candidate.get('location', ''),
            "linkedin": candidate.get('linkedin', ''),
            "skills": candidate.get('skills', []),
            "experience": candidate.get('experience', []),
            "education": candidate.get('education', [])
        }

    def _process_match_results(self, results) -> List[Dict[str, Any]]:
        """Process and format match results"""
        matches = []
        for match in results.matches:
            try:
                metadata = match.metadata
                matches.append({
                    'id': match.id,
                    'metadata': {
                        'name': metadata.get('name', ''),
                        'email': metadata.get('email', ''),
                        'phone': metadata.get('phone', ''),
                        'current_title': metadata.get('current_title', ''),
                        'current_company': metadata.get('current_company', ''),
                        'location': metadata.get('location', ''),
                        'linkedin': metadata.get('linkedin', ''),
                        'skills': metadata.get('skills', []),
                        'experience': metadata.get('experience', []),
                        'education': metadata.get('education', [])
                    },
                    'score': match.score
                })
            except Exception as e:
                logger.error(f"Error processing match result: {e}")
                continue
        return matches

    def _get_resume_embedding(self, resume: Resume) -> List[float]:
        """Get embedding for resume text"""
        try:
            text_parts = []
            if resume.text:
                text_parts.append(resume.text)

            if resume.skills:
                if isinstance(resume.skills, dict):
                    soft_skills = resume.skills.get('soft', [])
                    tech_skills = resume.skills.get('technical', [])
                    text_parts.extend([str(skill) for skill in soft_skills + tech_skills])
                else:
                    text_parts.extend([str(skill) for skill in resume.skills])

            if resume.experience:
                text_parts.extend([str(exp) for exp in resume.experience])

            combined_text = ' '.join(text_parts)
            return self._get_embedding(combined_text)
        except Exception as e:
            logger.error(f"Error getting resume embedding: {e}")
            raise

    def _get_job_embedding(self, job: Job) -> List[float]:
        """Get embedding for job text"""
        try:
            text_parts = []
            if job.title:
                text_parts.append(job.title)
            if job.description:
                text_parts.append(job.description)
            if job.requirements:
                text_parts.extend([str(req) for req in job.requirements])

            combined_text = ' '.join(text_parts)
            return self._get_embedding(combined_text)
        except Exception as e:
            logger.error(f"Error getting job embedding: {e}")
            raise

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise

    def clear_namespace(self, namespace: str) -> None:
        """Clear all vectors from a namespace"""
        try:
            logger.info(f"Clearing namespace: {namespace}")
            self.index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Successfully cleared namespace: {namespace}")
        except Exception as e:
            logger.error(f"Error clearing namespace: {e}")
            raise

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            if not vec1 or not vec2:
                logger.warning("Empty vectors provided for similarity calculation")
                return 0.0

            # Convert to numpy arrays
            a = np.array(vec1)
            b = np.array(vec2)

            # Calculate cosine similarity
            similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
