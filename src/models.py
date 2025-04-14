from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Job:
    title: str
    company: str
    description: str
    location: Optional[str] = None
    salary: Optional[str] = None
    requirements: Optional[List[str]] = None
    source: Optional[str] = None

@dataclass
class Resume:
    name: str
    email: str
    phone: str
    text: str
    skills: List[str]
    experience: List[str]
    education: List[str]

@dataclass
class MatchResult:
    job: Job
    score: float
    justification: str 