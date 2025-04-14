import pandas as pd
from loguru import logger
from typing import List
from src.models import (Job)
import os


class JobProcessor:
    def __init__(self):
        logger.info("Initializing JobProcessor")
        self.supported_formats = ['.xlsx', '.csv']
        logger.debug(f"Supported file formats: {self.supported_formats}")

    def process_file(self, file_path: str) -> List[Job]:
        """Process job file and return list of jobs"""
        try:
            logger.info(f"Processing job file: {file_path}")

            # Check file format
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.supported_formats:
                logger.error(f"Unsupported file format: {file_ext}")
                return []

            # Read file
            if file_ext == '.xlsx':
                df = pd.read_excel(file_path)
            else:  # .csv
                df = pd.read_csv(file_path)

            # Process each row
            jobs = []
            for _, row in df.iterrows():
                try:
                    # Create job object
                    job = Job(
                        title=row.get('title', ''),
                        company=row.get('company', ''),
                        location=row.get('location', ''),
                        salary=row.get('salary', ''),
                        description=row.get('description', ''),
                        requirements=row.get('requirements', [])
                    )
                    jobs.append(job)
                except Exception as e:
                    logger.error(f"Error processing job row: {e}")
                    continue

            logger.info(f"Successfully processed {len(jobs)} jobs from {file_path}")
            return jobs

        except Exception as e:
            logger.error(f"Error processing job file: {e}")
            return []

    def _parse_requirements(self, requirements_str: str) -> List[str]:
        """
        Parse requirements string into list of requirements
        """
        try:
            logger.debug("Parsing job requirements")
            if not requirements_str or pd.isna(requirements_str):
                logger.warning("Empty requirements string")
                return []

            # Split by common delimiters
            delimiters = [';', ',', '\n', '|']
            for delimiter in delimiters:
                if delimiter in requirements_str:
                    requirements = [req.strip() for req in requirements_str.split(delimiter) if req.strip()]
                    logger.debug(f"Found {len(requirements)} requirements using delimiter: {delimiter}")
                    return requirements

            # If no delimiter found, return as single requirement
            logger.debug("No delimiter found, treating as single requirement")
            return [requirements_str.strip()]
        except Exception as e:
            logger.error(f"Error parsing requirements: {e}")
            return []