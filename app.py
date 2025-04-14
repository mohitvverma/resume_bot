import streamlit as st
import os
import pandas as pd
import asyncio
from dotenv import load_dotenv
from src.resume_processor import ResumeProcessor
from src.job_processor import JobProcessor
from src.vector_store import VectorStore
from src.models import Resume, Job
from loguru import logger
from src.match_scorer import MatchScorer
import tempfile
from src.candidate_matcher import CandidateMatcher

# Load environment variables
load_dotenv()

# Create temp directory if it doesn't exist
temp_dir = "temp"
os.makedirs(temp_dir, exist_ok=True)
logger.info(f"Created/Verified temp directory at: {os.path.abspath(temp_dir)}")

# Initialize processors and vector store
resume_processor = ResumeProcessor()
job_processor = JobProcessor()
vector_store = VectorStore()
match_scorer = MatchScorer()
candidate_matcher = CandidateMatcher()  # Add persistent matcher instance


def save_uploaded_file(uploaded_file, temp_dir):
    """Save uploaded file to temporary directory"""
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def process_excel_file(file_path):
    """Process Excel/CSV file and return list of jobs"""
    try:
        # Read the file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        # Validate required columns
        required_columns = ['title', 'company', 'requirements', 'description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Process each row
        jobs = []
        for _, row in df.iterrows():
            job = Job(
                title=row['title'],
                company=row['company'],
                location=row.get('location', ''),
                salary=row.get('salary', ''),
                description=row['description'],
                requirements=row['requirements'].split(', ') if isinstance(row['requirements'], str) else []
            )
            jobs.append(job)

        return jobs
    except Exception as e:
        logger.error(f"Error processing Excel file: {e}")
        raise


def process_job_file(file):
    """Process job file and store in vector store"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

        jobs = job_processor.process_file(tmp_file_path)
        for job in jobs:
            # Generate job embedding with proper text
            job_text_parts = []
            if job.title:
                job_text_parts.append(job.title)
            if job.description:
                job_text_parts.append(job.description)
            if job.requirements:
                if isinstance(job.requirements, list):
                    job_text_parts.extend(job.requirements)
                else:
                    job_text_parts.append(job.requirements)

            job_text = ' '.join(job_text_parts)

            if not job_text.strip():
                logger.warning(f"Skipping job with empty text: {job.title} at {job.company}")
                continue

            job_embedding = resume_processor._get_embeddings(job_text)
            if not job_embedding:
                logger.warning(f"Skipping job due to failed embedding generation: {job.title} at {job.company}")
                continue

            vector_store.store_job(job, job_embedding)

        os.unlink(tmp_file_path)
        return True
    except Exception as e:
        logger.error(f"Error processing job file: {e}")
        return False


async def process_resumes(resume_files):
    """Process multiple resumes and find matches"""
    results = []
    for resume_file in resume_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(resume_file.name)[1]) as tmp_file:
                tmp_file.write(resume_file.getvalue())
                tmp_file_path = tmp_file.name

            resume = await resume_processor.process_resume(tmp_file_path)
            if resume:
                matches = vector_store.find_matches(resume)
                for match in matches:
                    # Get detailed match analysis
                    match_analysis = await match_scorer.score_match(
                        {
                            "title": match.get("job_title", "N/A"),
                            "company": match.get("company", "N/A"),
                            "description": match.get("job_description", "N/A"),
                            "requirements": match.get("requirements", "N/A"),
                            "location": match.get("location", "N/A"),
                            "salary": match.get("salary", "N/A")
                        },
                        resume.__dict__
                    )
                    results.append({
                        "resume_name": resume.name,
                        "job_title": match.get("job_title", "N/A"),
                        "company": match.get("company", "N/A"),
                        "match_score": match_analysis["match_score"],
                        "justification": match_analysis["justification"],
                        "key_matching_points": match_analysis["key_matching_points"],
                        "areas_for_improvement": match_analysis["areas_for_improvement"],
                        "job_description": match.get("job_description", "N/A"),
                        "job_requirements": match.get("requirements", "N/A"),
                        "resume_summary": resume.text[:500] + "..." if len(resume.text) > 500 else resume.text
                    })

            os.unlink(tmp_file_path)
        except Exception as e:
            logger.error(f"Error processing resume file: {e}")
            continue

    return results


async def find_candidate_matches(job: Job, csv_path: str, top_k: int = 5):
    """Find best matching candidates for a job"""
    try:
        logger.info(f"Finding matches for job: {job.title}")
        matcher = CandidateMatcher()

        # Load candidates
        candidates_df = matcher.load_candidates(csv_path)
        if candidates_df.empty:
            logger.error("No candidates loaded")
            return []

        # Find matches
        matches = await matcher.find_matches(job, candidates_df, top_k)

        # Generate LinkedIn messages
        for match in matches:
            match['linkedin_message'] = matcher.generate_linkedin_message(job, match)

        return matches
    except Exception as e:
        logger.error(f"Error finding candidate matches: {e}")
        return []


def main():
    st.set_page_config(
        page_title="Resume-Job Matcher",
        page_icon="🤝",
        layout="wide"
    )

    st.title("🤝 Resume-Job Matcher")

    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Job Data Ingestion", "Resume Matching", "Candidate Search"])

    with tab1:
        st.header("Step 1: Upload Job Data")
        st.markdown("""
        Upload Excel or CSV files containing job descriptions. Required columns:
        - title
        - company
        - description
        - requirements (comma-separated)
        Optional columns:
        - location
        - salary
        """)

        # Job Description Upload
        job_files = st.file_uploader(
            "Upload job description files (Excel, CSV)",
            type=["xlsx", "xls", "csv"],
            accept_multiple_files=True
        )

        if job_files:
            if st.button("Process Job Files"):
                with st.spinner("Processing job descriptions..."):
                    total_jobs = 0
                    for job_file in job_files:
                        try:
                            # Process the file
                            if process_job_file(job_file):
                                total_jobs += 1
                            else:
                                st.error(f"Error processing {job_file.name}: File processing failed")
                        except Exception as e:
                            st.error(f"Error processing {job_file.name}: {str(e)}")
                            logger.error(f"Error processing job file: {e}")

                    if total_jobs > 0:
                        st.success(f"✅ Successfully processed {total_jobs} jobs in total")

        # Clear Jobs Data
        if st.button("Clear All Jobs"):
            try:
                vector_store.clear_namespace("jobs")
                st.success("All jobs cleared successfully!")
            except Exception as e:
                st.error(f"Error clearing jobs: {str(e)}")
                logger.error(f"Error clearing jobs: {e}")

    with tab2:
        st.header("Step 2: Match Resume")
        st.markdown("""
        Upload one or more resumes to find matching jobs. The system will:
        1. Extract skills and experience from your resume
        2. Find matching jobs based on your profile
        3. Show detailed job matches with scores
        """)

        # Resume Upload
        resume_files = st.file_uploader(
            "Upload your resumes (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )

        if resume_files:
            if st.button("Find Matching Jobs"):
                with st.spinner("Processing resumes and finding matches..."):
                    try:
                        # Process resumes and get matches
                        results = asyncio.run(process_resumes(resume_files))

                        if results:
                            # Display results
                            st.header("Matching Results")

                            # Create summary table
                            summary_data = []
                            for result in results:
                                summary_data.append({
                                    "Resume": result['resume_name'],
                                    "Job Title": result['job_title'],
                                    "Company": result['company'],
                                    "Match Score": f"{result['match_score']}/10",
                                    "Key Skills": ", ".join(result['job_requirements']) if isinstance(
                                        result['job_requirements'], list) else result['job_requirements']
                                })

                            # Display summary table
                            st.subheader("Summary View")
                            st.dataframe(
                                pd.DataFrame(summary_data),
                                column_config={
                                    "Resume": "Resume",
                                    "Job Title": "Job Title",
                                    "Company": "Company",
                                    "Match Score": "Match Score",
                                    "Key Skills": "Required Skills"
                                },
                                hide_index=True,
                                use_container_width=True
                            )

                            st.markdown("---")

                            # Detailed View
                            st.subheader("Detailed Analysis")
                            for result in results:
                                st.subheader(f"Match for {result['resume_name']}")

                                # Job Details
                                st.markdown("### Job Details")
                                st.markdown(f"**Title:** {result['job_title']}")
                                st.markdown(f"**Company:** {result['company']}")
                                st.markdown(f"**Description:** {result['job_description']}")
                                st.markdown(f"**Requirements:** {result['job_requirements']}")

                                # Match Analysis
                                st.markdown("### Match Analysis")
                                st.markdown(f"**Match Score:** {result['match_score']}/10")
                                st.markdown(f"**Justification:** {result['justification']}")

                                # Key Matching Points
                                st.markdown("### Key Matching Points")
                                for point in result['key_matching_points']:
                                    st.markdown(f"- {point}")

                                # Areas for Improvement
                                st.markdown("### Areas for Improvement")
                                for area in result['areas_for_improvement']:
                                    st.markdown(f"- {area}")

                                # Resume Summary
                                st.markdown("### Resume Summary")
                                st.markdown(result['resume_summary'])

                                st.markdown("---")
                        else:
                            st.info("""
                            No matching jobs found. This could be because:
                            1. No jobs have been uploaded yet
                            2. Your skills don't match the available jobs
                            3. Try uploading more job descriptions in the Data Ingestion tab
                            """)

                    except Exception as e:
                        st.error(f"Error processing resume: {str(e)}")
                        logger.error(f"Error processing resume: {e}")

    with tab3:
        st.header("Step 3: Candidate Search")
        st.markdown("""
        1. First, ingest candidate data (this will clear any existing data)
        2. Then, search for matching candidates based on job details
        """)

        # Step 1: Data Ingestion
        st.subheader("1. Data Ingestion")

        # Toggle for data ingestion
        ingest_new_data = st.toggle(
            "Enable Data Ingestion",
            value=False,
            help="Toggle ON to ingest new candidate data. This will clear any existing data."
        )

        if ingest_new_data:
            candidates_file = st.file_uploader(
                "Upload candidate database (CSV)",
                type=["csv"],
                help="Upload the Juicebox export CSV file containing candidate data."
            )

            if candidates_file:
                # Show the file details
                st.info(f"File uploaded: {candidates_file.name}")

                # Add Process button
                if st.button("Process Data", type="primary"):
                    with st.spinner("Processing candidate data..."):
                        try:
                            # Save uploaded file
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                                tmp_file.write(candidates_file.getvalue())
                                csv_path = tmp_file.name

                            # Clear existing data from vector store
                            vector_store.clear_namespace(vector_store.namespaces["candidate_search"])
                            logger.info("Cleared existing candidate data")

                            # Load and store new candidates
                            df = candidate_matcher.load_candidates(csv_path, ingest_new_data=True)

                            # Clean up temp file
                            os.unlink(csv_path)

                            if not df.empty:
                                st.success(f"✅ Successfully ingested {len(df)} candidates")

                                # Display candidate summary
                                st.markdown("### Database Summary")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Candidates", len(df))
                                with col2:
                                    companies = df['Current Org Name'].nunique()
                                    st.metric("Unique Companies", companies)
                                with col3:
                                    titles = df['Current Title'].nunique()
                                    st.metric("Unique Titles", titles)

                                # Show sample of the data
                                st.markdown("### Sample Candidates")
                                sample_df = df[
                                    ['First name', 'Last name', 'Current Title', 'Current Org Name', 'Location']].head()
                                st.dataframe(sample_df, use_container_width=True)
                            else:
                                st.error("Failed to load candidate data. Please check the file format and try again.")
                        except Exception as e:
                            st.error(f"Error processing candidate database: {str(e)}")
                            logger.error(f"Error processing candidate database: {e}")
            else:
                st.info("👆 Please upload your candidate database (CSV) file to proceed.")
        else:
            st.info("Toggle ON 'Enable Data Ingestion' to upload new candidate data.")

        # Step 2: Candidate Search
        st.subheader("2. Candidate Search")

        # Toggle for search functionality
        enable_search = st.toggle(
            "Enable Search",
            value=False,
            help="Toggle ON to search for matching candidates"
        )

        if enable_search:
            # Check if we have data to search against
            if not candidate_matcher._check_candidates_stored():
                st.warning("No candidate data available. Please ingest data first.")
            else:
                with st.form("job_details_form"):
                    job_title = st.text_input("Job Title", "Senior Software Engineer")
                    job_company = st.text_input("Company", "Probook AI")
                    job_description = st.text_area("Job Description",
                                                   "Looking for a senior software engineer with experience in Python, ML, and cloud technologies.")
                    job_location = st.text_input("Location", "San Francisco")
                    job_salary = st.text_input("Salary Range", "$150,000 - $200,000")
                    job_requirements = st.text_area("Requirements (one per line)",
                                                    "Python\nMachine Learning\nAWS\nDocker\nKubernetes")

                    submitted = st.form_submit_button("Find Matching Candidates")

                    if submitted:
                        with st.spinner("Finding matches..."):
                            try:
                                # Create job object
                                job = Job(
                                    title=job_title,
                                    company=job_company,
                                    description=job_description,
                                    location=job_location,
                                    salary=job_salary,
                                    requirements=[req.strip() for req in job_requirements.split('\n') if req.strip()],
                                    source="user_input"
                                )

                                # Find matches using the persistent matcher
                                matches = asyncio.run(candidate_matcher.find_matches(job))

                                if matches:
                                    # Step 3: Display Results
                                    st.subheader("3. Matching Results")

                                    # Create summary table
                                    summary_data = []
                                    for match in matches:
                                        summary_data.append({
                                            "Name": match['name'],
                                            "Current Title": match['current_title'],
                                            "Company": match['current_company'],
                                            "Match Score": f"{match['match_score']}/10",
                                            "LinkedIn": match['linkedin']
                                        })

                                    st.dataframe(
                                        pd.DataFrame(summary_data),
                                        column_config={
                                            "Name": "Name",
                                            "Current Title": "Current Title",
                                            "Company": "Company",
                                            "Match Score": "Match Score",
                                            "LinkedIn": st.column_config.LinkColumn("LinkedIn Profile")
                                        },
                                        hide_index=True,
                                        use_container_width=True
                                    )

                                    # Detailed candidate cards
                                    st.markdown("### Detailed Analysis")
                                    for match in matches:
                                        with st.expander(
                                                f"{match['name']} - {match['current_title']} at {match['current_company']}"):
                                            col1, col2 = st.columns([2, 1])
                                            with col1:
                                                st.markdown(f"**Match Score:** {match['match_score']}/10")
                                                st.markdown(
                                                    f"**Current Role:** {match['current_title']} at {match['current_company']}")
                                                st.markdown(
                                                    f"**Contact:** [LinkedIn]({match['linkedin']}) | {match['email']} | {match['phone']}")
                                                st.markdown("**Key Matching Points:**")
                                                for point in match['key_matching_points']:
                                                    st.markdown(f"- {point}")
                                                st.markdown("**Areas for Improvement:**")
                                                for area in match['areas_for_improvement']:
                                                    st.markdown(f"- {area}")
                                            with col2:
                                                st.markdown("**Suggested LinkedIn Message:**")
                                                message = candidate_matcher.generate_linkedin_message(job, match)
                                                st.text_area("", message, height=150, key=f"msg_{match['name']}")
                                                st.caption(f"{len(message)}/250 characters")
                                else:
                                    st.warning("No matching candidates found. Try adjusting the job requirements.")

                            except Exception as e:
                                st.error(f"Error finding matches: {str(e)}")
                                logger.error(f"Error in candidate matching: {e}")
        else:
            st.info("Toggle ON 'Enable Search' to search for matching candidates.")


if __name__ == "__main__":
    main()