import asyncio
import re

import pandas as pd

from loguru import logger
from typing import List, Dict, Any
from src.models import Job
from src.vector_store import VectorStore
from concurrent.futures import ThreadPoolExecutor
from src.match_scorer import MatchScorer


class CandidateMatcher:
    def __init__(self):
        logger.info("Initializing CandidateMatcher")
        self.vector_store = VectorStore()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.candidates_df = None  # Store the loaded DataFrame
        self.match_scorer = MatchScorer()  # Initialize MatchScorer
        self._load_existing_data()  # Try to load existing data on initialization

    def _load_existing_data(self) -> None:
        """Try to load existing data from vector store"""
        try:
            # Check if candidates exist in vector store
            if self._check_candidates_stored():
                logger.info("Found existing candidates in vector store")
                # Query to get all candidates
                results = self.vector_store.index.query(
                    vector=[0.0] * 1536,  # Dummy vector
                    top_k=1000,  # Adjust based on expected number of candidates
                    namespace=self.vector_store.namespaces["candidate_search"],
                    include_metadata=True
                )

                # Convert results to DataFrame
                candidates_data = []
                for match in results.matches:
                    metadata = match.metadata
                    candidates_data.append({
                        'First name': metadata.get('name', '').split()[0] if metadata.get('name') else '',
                        'Last name': metadata.get('name', '').split()[1] if metadata.get('name') else '',
                        'Email': metadata.get('email', ''),
                        'Phone': metadata.get('phone', ''),
                        'Current Title': metadata.get('current_title', ''),
                        'Current Org Name': metadata.get('current_company', ''),
                        'Location': metadata.get('location', ''),
                        'LinkedIn Profile': metadata.get('linkedin', ''),
                        'Skills': metadata.get('skills', []),
                        'Experience': metadata.get('experience', []),
                        'Education': metadata.get('education', [])
                    })

                if candidates_data:
                    self.candidates_df = pd.DataFrame(candidates_data)
                    logger.info(f"Loaded {len(self.candidates_df)} existing candidates")
            else:
                logger.info("No existing candidates found in vector store")
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")

    def load_candidates(self, csv_path: str = None, ingest_new_data: bool = True) -> pd.DataFrame:
        """Load candidates from CSV file or use existing data"""
        try:
            if csv_path and ingest_new_data:
                logger.info(f"Loading new candidates from {csv_path}")
                df = pd.read_csv(csv_path)

                # Store candidates in vector store
                asyncio.run(self._store_candidates(df))

                # Store the DataFrame for later use
                self.candidates_df = df

                return df
            elif not ingest_new_data:
                # Try to use existing data
                logger.info("Using existing candidate data")
                if self.candidates_df is not None and not self.candidates_df.empty:
                    return self.candidates_df
                else:
                    logger.warning("No existing candidate data found")
                    return pd.DataFrame()
            else:
                logger.error("No candidate data available")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading candidates: {e}")
            return pd.DataFrame()

    def _check_candidates_stored(self) -> bool:
        """Check if candidates are already stored in the vector store"""
        try:
            # Try to query the candidates namespace
            results = self.vector_store.index.query(
                vector=[0.0] * 1536,  # Dummy vector
                top_k=1,
                namespace=self.vector_store.namespaces["candidate_search"],
                include_metadata=True
            )
            return len(results.matches) > 0
        except Exception as e:
            logger.warning(f"Error checking stored candidates: {e}")
            return False

    async def _store_candidates(self, candidates_df: pd.DataFrame) -> None:
        """Store candidates in vector store"""
        try:
            logger.info(f"Storing {len(candidates_df)} candidates in vector store")

            # Add ID column if it doesn't exist
            if 'id' not in candidates_df.columns:
                candidates_df['id'] = [f"candidate_{i + 1}" for i in range(len(candidates_df))]

            # Process candidates in parallel
            with ThreadPoolExecutor() as executor:
                futures = []
                for _, row in candidates_df.iterrows():
                    try:
                        # Create candidate dictionary
                        candidate = {
                            'id': str(row['id']),
                            'name': f"{row.get('First name', '')} {row.get('Last name', '')}".strip(),
                            'email': row.get('Email', ''),
                            'phone': row.get('Phone', ''),
                            'current_title': row.get('Current Title', ''),
                            'current_company': row.get('Current Org Name', ''),
                            'location': row.get('Location', ''),
                            'linkedin': row.get('LinkedIn Profile', ''),
                            'skills': row.get('Skills', []),
                            'experience': row.get('Experience', []),
                            'education': row.get('Education', [])
                        }

                        # Generate embedding for candidate
                        candidate_text = self._create_candidate_text(candidate)
                        embedding = self.vector_store._get_embedding(candidate_text)

                        # Store candidate in vector store
                        future = executor.submit(
                            self.vector_store.store_candidate,
                            candidate,
                            embedding
                        )
                        futures.append(future)

                        # Log progress every 10 candidates
                        if len(futures) % 10 == 0:
                            logger.info(f"Processed {len(futures)} candidates")

                    except Exception as e:
                        logger.error(f"Error processing candidate {row.get('id', 'unknown')}: {e}")
                        continue

                # Wait for all futures to complete
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error storing candidate: {e}")
                        continue

            logger.info(f"Successfully stored {len(futures)} candidates in vector store")

        except Exception as e:
            logger.error(f"Error storing candidates: {e}")
            raise

    def _store_single_candidate(self, row: pd.Series) -> None:
        """Store a single candidate in the vector store"""
        try:
            # Create candidate text for embedding
            candidate_text = self._create_candidate_text(row)

            # Get embedding
            embedding = self.vector_store._get_embedding(candidate_text)

            # Create metadata dictionary
            metadata = {
                'id': str(row['id']),
                'name': f"{row.get('First name', '')} {row.get('Last name', '')}",
                'current_title': row.get('Current Title', ''),
                'current_company': row.get('Current Org Name', ''),
                'email': row.get('Email', ''),
                'phone': row.get('Phone', ''),
                'linkedin': row.get('LinkedIn Profile', ''),
                'skills': row.get('Skills', ''),
                'experience': row.get('Experience', ''),
                'education': row.get('Education', '')
            }

            # Store in vector store with proper ID
            vector_id = f"candidate_{row['id']}"
            self.vector_store.index.upsert(
                vectors=[(vector_id, embedding, metadata)],
                namespace="candidates"
            )

        except Exception as e:
            logger.error(f"Error storing candidate {row.get('First name', 'Unknown')}: {e}")
            raise

    def _create_candidate_text(self, row: pd.Series) -> str:
        """Create text for candidate embedding"""
        text_parts = []

        # Add basic information
        if 'Current Title' in row:
            text_parts.append(str(row['Current Title']))
        if 'Current Org Name' in row:
            text_parts.append(str(row['Current Org Name']))
        if 'Skills' in row:
            text_parts.append(str(row['Skills']))
        if 'Experience' in row:
            text_parts.append(str(row['Experience']))
        if 'Education' in row:
            text_parts.append(str(row['Education']))

        return ' '.join(text_parts)

    async def find_matches(self, job: Job, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find matching candidates for a job"""
        try:
            logger.info(f"Finding matches for job: {job.title}")

            # Find matches in vector store
            matches = self.vector_store.find_candidates(job, top_k)

            if not matches:
                logger.warning("No matches found")
                return []

            # Process matches
            processed_matches = []
            for match in matches:
                try:
                    # Get metadata from match
                    metadata = match.get('metadata', {})

                    # Get detailed match analysis
                    match_analysis = await self.match_scorer.score_match(
                        job.__dict__,
                        metadata
                    )

                    # Create processed match with all available fields
                    processed_match = {
                        'id': match.get('id', ''),
                        'name': metadata.get('name', ''),
                        'email': metadata.get('email', ''),
                        'phone': metadata.get('phone', ''),
                        'current_title': metadata.get('current_title', ''),
                        'current_company': metadata.get('current_company', ''),
                        'linkedin': metadata.get('linkedin', ''),
                        'match_score': match_analysis.get('match_score', 0),
                        'key_matching_points': match_analysis.get('key_matching_points', []),
                        'areas_for_improvement': match_analysis.get('areas_for_improvement', [])
                    }

                    # Only add match if we have at least a name
                    if processed_match['name']:
                        processed_matches.append(processed_match)
                    else:
                        logger.warning(f"Skipping match with missing name: {match.get('id', 'unknown')}")

                except Exception as e:
                    logger.error(f"Error processing match: {e}")
                    continue

            return processed_matches

        except Exception as e:
            logger.error(f"Error finding matches: {e}")
            return []

    def _get_matching_points(self, job: Job, candidate: pd.Series) -> List[str]:
        """Get key matching points between job and candidate"""
        matching_points = []

        # Check title match
        if 'Current Title' in candidate and job.title.lower() in candidate['Current Title'].lower():
            matching_points.append(f"Current title matches job title")

        # Check skills match
        if 'Skills' in candidate:
            candidate_skills = set(str(candidate['Skills']).lower().split(','))
            job_skills = set(skill.lower() for skill in job.requirements)
            matching_skills = candidate_skills.intersection(job_skills)
            if matching_skills:
                matching_points.append(f"Skills match: {', '.join(matching_skills)}")

        # Check experience match
        if 'Experience' in candidate and job.description:
            if job.description.lower() in str(candidate['Experience']).lower():
                matching_points.append("Experience matches job description")

        return matching_points

    def _get_improvement_areas(self, job: Job, candidate: pd.Series) -> List[str]:
        """Get areas where candidate could improve"""
        improvement_areas = []

        # Check missing skills
        if 'Skills' in candidate:
            candidate_skills = set(str(candidate['Skills']).lower().split(','))
            job_skills = set(skill.lower() for skill in job.requirements)
            missing_skills = job_skills - candidate_skills
            if missing_skills:
                improvement_areas.append(f"Missing skills: {', '.join(missing_skills)}")

        # Check experience gaps
        if 'Experience' in candidate and job.description:
            if job.description.lower() not in str(candidate['Experience']).lower():
                improvement_areas.append("Experience could better match job description")

        return improvement_areas

    def generate_linkedin_message(self, job: Job, candidate: Dict[str, Any]) -> str:
        """Generate personalized LinkedIn message"""
        try:
            message = f"""Hi {candidate['name'].split()[0]},

    I came across your profile and noticed your experience in {candidate['current_title']} at {candidate['current_company']}. We're currently looking for a {job.title} at {job.company} and I think your background could be a great fit.

    The role involves {job.description[:100]}... Would you be open to a quick chat about this opportunity?

    Best regards,
    [Your Name]
    """

            return message
        except Exception as e:
            logger.error(f"Error generating LinkedIn message: {e}")
            return ""
