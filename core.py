"""
Core Assessment Engine: Taxonomy, Question Bank, Extraction, RAG, Blueprinting & Simulation.
Shared domain module utilized by both FastAPI backend and Streamlit UI.
"""

import os
import re
import json
import time
import random
from io import BytesIO
from collections import Counter, defaultdict

try:
    import requests
except ImportError:
    requests = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


def load_env_file():
    """Loads key-value pairs from .env in workspace root if present."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and (k not in os.environ or not os.environ[k]):
                            os.environ[k] = v
        except Exception:
            pass

load_env_file()


TAXONOMY = {
    "Python": {
        "canonical": "Python",
        "category": "Technical",
        "aliases": ["python", "pandas", "numpy", "pytest", "fastapi", "django", "flask", "pydantic", "asyncio"],
    },
    "Java": {
        "canonical": "Java",
        "category": "Technical",
        "aliases": ["java", "spring boot", "spring", "junit", "maven", "hibernate", "jvm"],
    },
    "SQL & Databases": {
        "canonical": "SQL & Databases",
        "category": "Technical",
        "aliases": ["sql", "mysql", "postgresql", "database", "query optimization", "oracle", "nosql", "mongodb", "redis"],
    },
    "REST APIs & Web Services": {
        "canonical": "REST APIs & Web Services",
        "category": "Technical",
        "aliases": ["rest api", "restful", "http api", "api development", "json", "graphql", "endpoints"],
    },
    "Problem Solving & DSA": {
        "canonical": "Problem Solving & DSA",
        "category": "Technical",
        "aliases": ["data structures", "algorithms", "problem solving", "complexity", "leetcode", "dynamic programming"],
    },
    "Cloud & DevOps": {
        "canonical": "Cloud & DevOps",
        "category": "Tools",
        "aliases": ["aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "jenkins", "terraform", "github actions"],
    },
    "Git & Code Collaboration": {
        "canonical": "Git & Code Collaboration",
        "category": "Tools",
        "aliases": ["git", "github", "version control", "code review", "pull request", "branching", "gitlab"],
    },
    "Testing & Quality Assurance": {
        "canonical": "Testing & Quality Assurance",
        "category": "Technical",
        "aliases": ["unit testing", "integration testing", "pytest", "junit", "test automation", "selenium", "tdd", "mocking"],
    },
    "Frontend Development": {
        "canonical": "Frontend Development",
        "category": "Technical",
        "aliases": ["html", "css", "javascript", "react", "frontend", "front-end", "typescript", "vue"],
    },
    "System Design & Architecture": {
        "canonical": "System Design & Architecture",
        "category": "Technical",
        "aliases": ["system design", "scalability", "distributed systems", "architecture", "high availability", "microservices"],
    },
    "Security & Authentication": {
        "canonical": "Security & Authentication",
        "category": "Technical",
        "aliases": ["authentication", "authorization", "oauth", "oidc", "jwt", "security", "encryption", "owasp"],
    },
    "Data Analysis & Visualization": {
        "canonical": "Data Analysis & Visualization",
        "category": "Technical",
        "aliases": ["data analysis", "power bi", "tableau", "excel", "visualization", "statistics", "eda"],
    },
    "Stakeholder Communication": {
        "canonical": "Stakeholder Communication",
        "category": "Behavioral",
        "aliases": ["communication", "stakeholder", "presentation", "cross-functional", "verbal", "written communication"],
    },
    "Leadership & Mentorship": {
        "canonical": "Leadership & Mentorship",
        "category": "Behavioral",
        "aliases": ["leadership", "mentor", "mentoring", "lead a team", "people management", "coaching", "initiative"],
    },
    "Agile & Ownership": {
        "canonical": "Agile & Ownership",
        "category": "Behavioral",
        "aliases": ["agile", "scrum", "sprint", "ownership", "accountability", "self-starter", "fast-paced"],
    },
}

QUESTION_BANK = [
    {
        "id": "Q-PY-01",
        "competency": "Python",
        "subskill": "Core Data Structures & Mutability",
        "difficulty": "Easy",
        "cognitive_level": "Remember/Understand",
        "job_relevance": "Direct Core Requirement",
        "type": "Technical",
        "question": "Explain the behavioral and memory differences between a list and a tuple in Python. When would you strictly prefer a tuple?",
        "expected": "Lists are mutable and over-allocate memory; tuples are immutable, memory-efficient, and hashable (usable as dictionary keys).",
        "rubric": "Full marks for mutability distinction, memory implications, and hashability use-cases.",
        "minutes": 4,
    },
    {
        "id": "Q-PY-02",
        "competency": "Python",
        "subskill": "Profiling & Concurrency / Memory",
        "difficulty": "Medium",
        "cognitive_level": "Apply/Analyze",
        "job_relevance": "Direct Core Requirement",
        "type": "Scenario",
        "question": "A production Python service processing incoming 500MB CSV files runs out of memory on Kubernetes. How do you re-architect the parsing pipeline?",
        "expected": "Stream file in chunks (e.g., pandas chunksize or generator streaming), avoid loading entire file into memory, use generators, profile with memory_profiler.",
        "rubric": "Demonstrates streaming/chunking awareness, memory profiling techniques, and resource cleanup.",
        "minutes": 8,
    },
    {
        "id": "Q-PY-03",
        "competency": "Python",
        "subskill": "AsyncIO & High Throughput I/O",
        "difficulty": "Hard",
        "cognitive_level": "Evaluate/Synthesize",
        "job_relevance": "Architecture & Scaling",
        "type": "Scenario",
        "question": "In a FastAPI service using AsyncIO, CPU-intensive data transformations cause HTTP request latencies to spike. Explain why and how to isolate the blocking calls.",
        "expected": "AsyncIO event loop is single-threaded; CPU-bound tasks block the event loop. Solution: offload CPU tasks to ProcessPoolExecutor, Celery, or background worker threads using run_in_executor.",
        "rubric": "Must correctly explain GIL/event loop blocking and provide process pool or external worker architecture.",
        "minutes": 10,
    },
    {
        "id": "Q-JV-01",
        "competency": "Java",
        "subskill": "OOP & JVM Memory Basics",
        "difficulty": "Easy",
        "cognitive_level": "Remember/Understand",
        "job_relevance": "Direct Core Requirement",
        "type": "Technical",
        "question": "Explain how Java manages Stack vs Heap memory and where object instances and reference variables reside.",
        "expected": "Stack holds method call frames and local primitive variables/references. Heap holds actual object instances managed by Garbage Collection.",
        "rubric": "Accurate distinction of Heap vs Stack lifetimes and Garbage Collection triggers.",
        "minutes": 4,
    },
    {
        "id": "Q-JV-02",
        "competency": "Java",
        "subskill": "Spring Boot Troubleshooting",
        "difficulty": "Medium",
        "cognitive_level": "Apply/Analyze",
        "job_relevance": "Operational Tooling",
        "type": "Scenario",
        "question": "A Spring Boot microservice periodically throws intermittent ConnectionPoolTimeoutException under peak traffic. Detail your investigation steps.",
        "expected": "Check HikariCP connection pool settings, inspect long-running transactions, ensure connections and entity managers are properly closed, check slow queries and thread dumps.",
        "rubric": "Systematic root cause analysis: connection pool tuning, unclosed connections, and DB query latency.",
        "minutes": 7,
    },
    {
        "id": "Q-SQL-01",
        "competency": "SQL & Databases",
        "subskill": "Relational Joins & Set Operations",
        "difficulty": "Easy",
        "cognitive_level": "Remember/Understand",
        "job_relevance": "Direct Core Requirement",
        "type": "Technical",
        "question": "Compare INNER JOIN, LEFT OUTER JOIN, and FULL OUTER JOIN with real-world nullability scenarios.",
        "expected": "INNER returns intersection. LEFT preserves left rows with null fillers. FULL preserves all rows from both sides.",
        "rubric": "Accurate Venn interpretation and null handling examples.",
        "minutes": 4,
    },
    {
        "id": "Q-SQL-02",
        "competency": "SQL & Databases",
        "subskill": "Query Optimization & Indexing",
        "difficulty": "Medium",
        "cognitive_level": "Apply/Analyze",
        "job_relevance": "Direct Core Requirement",
        "type": "Scenario",
        "question": "A PostgreSQL query filtering on status and created_at takes 12 seconds on a table with 25 million rows. How do you analyze EXPLAIN ANALYZE and design an optimal index?",
        "expected": "Run EXPLAIN (ANALYZE, BUFFERS); look for Seq Scan vs Index Scan; create composite B-tree index on (status, created_at) or partial index if status is skewed.",
        "rubric": "Correct index column ordering rationale, understanding of selectivity and scan types.",
        "minutes": 8,
    },
    {
        "id": "Q-SQL-03",
        "competency": "SQL & Databases",
        "subskill": "Transactions & Isolation Levels",
        "difficulty": "Hard",
        "cognitive_level": "Evaluate/Synthesize",
        "job_relevance": "Architecture & Scaling",
        "type": "Technical",
        "question": "Define Phantom Reads and explain which ANSI SQL isolation level prevents them. Compare Optimistic vs Pessimistic locking in high-concurrency inventory deductions.",
        "expected": "Phantom read occurs when range query returns different rows after another transaction commits. Prevented by Serializable (or Repeatable Read with MVCC gap locks). Optimistic uses version column; Pessimistic uses SELECT FOR UPDATE.",
        "rubric": "Clear explanation of isolation levels, locks, and trade-offs.",
        "minutes": 10,
    },
    {
        "id": "Q-API-01",
        "competency": "REST APIs & Web Services",
        "subskill": "HTTP Semantics & Status Codes",
        "difficulty": "Easy",
        "cognitive_level": "Remember/Understand",
        "job_relevance": "Direct Core Requirement",
        "type": "Technical",
        "question": "Explain the difference between idempotent and safe HTTP methods. Categorize GET, POST, PUT, PATCH, and DELETE.",
        "expected": "Safe methods do not mutate state (GET, HEAD). Idempotent methods have identical effects regardless of repeat count (GET, PUT, DELETE). POST is neither safe nor idempotent.",
        "rubric": "Full credit for correct classification and idempotency definition.",
        "minutes": 4,
    },
    {
        "id": "Q-API-02",
        "competency": "REST APIs & Web Services",
        "subskill": "Idempotent API Design",
        "difficulty": "Medium",
        "cognitive_level": "Apply/Analyze",
        "job_relevance": "Direct Core Requirement",
        "type": "Scenario",
        "question": "How do you design a payment transfer endpoint `POST /transfers` so that network retries never charge the customer twice?",
        "expected": "Require an Idempotency-Key header from client, store key in cache/DB with atomic check-and-set, return cached result if key exists, enforce expiration TTL.",
        "rubric": "Covers Idempotency-Key pattern, locking/atomic state, and caching strategies.",
        "minutes": 7,
    },
    {
        "id": "Q-DSA-01",
        "competency": "Problem Solving & DSA",
        "subskill": "Array Traversal & Complexity",
        "difficulty": "Easy",
        "cognitive_level": "Apply/Analyze",
        "job_relevance": "Direct Core Requirement",
        "type": "Coding",
        "question": "Write an algorithm to find the single non-duplicate number in an array where every other element appears twice. Target O(n) time and O(1) space.",
        "expected": "Use bitwise XOR across all elements. Identical numbers cancel out (x ^ x = 0), leaving the unique element.",
        "rubric": "Optimal XOR solution with time and space complexity explanation.",
        "minutes": 6,
    },
    {
        "id": "Q-DSA-02",
        "competency": "Problem Solving & DSA",
        "subskill": "Sliding Window & Two Pointers",
        "difficulty": "Medium",
        "cognitive_level": "Apply/Analyze",
        "job_relevance": "Direct Core Requirement",
        "type": "Coding",
        "question": "Given a string `s`, find the length of the longest substring without repeating characters. Walk through the algorithm and pointer movements.",
        "expected": "Sliding window with hash map storing last seen index of characters. Window expands right; when duplicate encountered, advance left pointer past previous duplicate index. O(n) time, O(min(m, n)) space.",
        "rubric": "Clean window manipulation, edge cases (empty string, all identical characters), complexity.",
        "minutes": 10,
    },
    {
        "id": "Q-SYS-01",
        "competency": "System Design & Architecture",
        "subskill": "Distributed Rate Limiting",
        "difficulty": "Hard",
        "cognitive_level": "Evaluate/Synthesize",
        "job_relevance": "Architecture & Scaling",
        "type": "System Design",
        "question": "Design a distributed rate limiter for an API gateway supporting 100,000 requests/sec across 15 microservices. Discuss Token Bucket vs Leaky Bucket and Redis clustering.",
        "expected": "Use Token Bucket with Redis and Lua script for atomic token decrement, sliding window counter for smooth limits, local memory caching with async sync to avoid Redis bottleneck.",
        "rubric": "Algorithm selection, atomic Lua execution, latency reduction, and failure fallback modes.",
        "minutes": 15,
    },
    {
        "id": "Q-CLD-01",
        "competency": "Cloud & DevOps",
        "subskill": "Containerization & Security",
        "difficulty": "Easy",
        "cognitive_level": "Remember/Understand",
        "job_relevance": "Operational Tooling",
        "type": "Technical",
        "question": "What are multi-stage Docker builds and what security and size advantages do they provide in a CI/CD pipeline?",
        "expected": "Multi-stage builds separate build dependencies (compilers, SDKs) from runtime image. Produces tiny final images with minimal attack surface.",
        "rubric": "Explains image size reduction, absence of build secrets/tools in production image.",
        "minutes": 4,
    },
    {
        "id": "Q-CLD-02",
        "competency": "Cloud & DevOps",
        "subskill": "Kubernetes Deployment Strategies",
        "difficulty": "Medium",
        "cognitive_level": "Apply/Analyze",
        "job_relevance": "Operational Tooling",
        "type": "Scenario",
        "question": "Compare Canary vs Blue-Green deployment strategies in Kubernetes. When would database migration constraints prevent pure Blue-Green?",
        "expected": "Blue-Green switches 100% traffic instantly; Canary gradually shifts traffic (5%, 25%, 100%). Breaking DB schema migrations that are not backwards-compatible break simultaneously active blue/green environments.",
        "rubric": "Traffic routing mechanisms, rollback speed, and backward-compatible DB expansion/contract pattern.",
        "minutes": 8,
    },
    {
        "id": "Q-BEH-01",
        "competency": "Stakeholder Communication",
        "subskill": "Cross-Functional Alignment",
        "difficulty": "Medium",
        "cognitive_level": "Apply/Analyze",
        "job_relevance": "Workplace Collaboration",
        "type": "Behavioral",
        "question": "A non-technical product manager demands a critical feature release in 2 days, but testing reveals severe edge-case crashes. How do you handle the discussion?",
        "expected": "Present trade-offs transparently without jargon, outline customer impact and risk, offer compromised scope (de-scoped MVP or feature flag release), and agree on a remediation plan.",
        "rubric": "Constructive conflict resolution, risk articulation, empathy, and data-driven options.",
        "minutes": 6,
    },
    {
        "id": "Q-BEH-02",
        "competency": "Leadership & Mentorship",
        "subskill": "Code Review & Team Enablement",
        "difficulty": "Medium",
        "cognitive_level": "Apply/Analyze",
        "job_relevance": "Team Enablement",
        "type": "Behavioral",
        "question": "A junior engineer repeatedly submits pull requests with substandard test coverage and architectural violations. How do you guide them constructively?",
        "expected": "1-on-1 private sync, understand their blockers/knowledge gaps, provide clear rubrics/examples, pair program on a sample module, set incremental goals and acknowledge improvements.",
        "rubric": "Mentorship approach, psychological safety, actionable feedback, and structured follow-up.",
        "minutes": 6,
    },
    {
        "id": "Q-SEC-01",
        "competency": "Security & Authentication",
        "subskill": "OAuth2 & JWT Token Validation",
        "difficulty": "Medium",
        "cognitive_level": "Apply/Analyze",
        "job_relevance": "Direct Core Requirement",
        "type": "Scenario",
        "question": "Walk through the required steps an API gateway must take to securely validate a stateless JWT bearer token before forwarding requests downstream.",
        "expected": "Verify signature against public JWKS key, verify algorithm isn't 'none', validate claims: expiration (exp), not-before (nbf), issuer (iss), audience (aud), and check revocation blocklist if supported.",
        "rubric": "Comprehensive security checks (signature, claims, key rotation, algorithms).",
        "minutes": 8,
    },
    {
        "id": "Q-TST-01",
        "competency": "Testing & Quality Assurance",
        "subskill": "Test Pyramid & Mocking Strategies",
        "difficulty": "Medium",
        "cognitive_level": "Apply/Analyze",
        "job_relevance": "Direct Core Requirement",
        "type": "Scenario",
        "question": "A developer writes unit tests where every database call, internal utility, and third-party API is heavily mocked. What pitfalls arise from over-mocking?",
        "expected": "Tests test the mock implementation rather than real system contracts; tests become brittle and break upon refactoring; false confidence because integration and DB constraints are untested.",
        "rubric": "Test pyramid balance, contract testing necessity, and distinction between unit and integration validation.",
        "minutes": 7,
    },
    {
        "id": "Q-GIT-01",
        "competency": "Git & Code Collaboration",
        "subskill": "Branching & Git Rebase vs Merge",
        "difficulty": "Easy",
        "cognitive_level": "Remember/Understand",
        "job_relevance": "Operational Tooling",
        "type": "Technical",
        "question": "When should you use `git rebase` versus `git merge`? What danger exists when rebasing shared remote branches?",
        "expected": "Use rebase on local feature branches for linear history; use merge for preserving branch topology on mainline. Rebasing rewritten commits on shared branches causes duplicate commits and merge chaos for teammates.",
        "rubric": "Commit history cleanliness, risk of rewriting shared history, and golden rule of rebasing.",
        "minutes": 5,
    },
]


def extract_text_from_bytes(filename: str, raw_bytes: bytes) -> str:
    """Extracts raw text from uploaded TXT, PDF, or DOCX bytes."""
    name = filename.lower()
    if name.endswith(".txt"):
        return raw_bytes.decode("utf-8", errors="replace")
    if name.endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError("pypdf is required to read PDF files")
        reader = PdfReader(BytesIO(raw_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".docx"):
        if Document is None:
            raise RuntimeError("python-docx is required to read DOCX files")
        doc = Document(BytesIO(raw_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError("Supported file formats: .txt, .pdf, .docx")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def detect_seniority(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(lead|architect|principal|staff|director|head of)\b", t):
        return "Lead"
    if re.search(r"\b(senior|sr\.?|5\+|6\+|7\+|8\+|10\+ years)\b", t):
        return "Senior"
    if re.search(r"\b(mid|intermediate|3-5 years|2-4 years)\b", t):
        return "Mid"
    if re.search(r"\b(junior|jr\.?|1-2 years|graduate|associate)\b", t):
        return "Junior"
    return "Mid"


def rule_based_extraction(jd_text: str) -> dict:
    norm = normalize_text(jd_text)
    seniority = detect_seniority(jd_text)
    
    sentences = re.split(r"(?<=[.!?])\s+|\n+", jd_text)
    responsibilities = []
    resp_keywords = ["responsible", "design", "develop", "maintain", "lead", "build", "collaborate", "ensure", "deliver", "oversee"]
    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) > 25 and any(w in s_clean.lower() for w in resp_keywords):
            if s_clean not in responsibilities:
                responsibilities.append(s_clean)
        if len(responsibilities) >= 5:
            break

    found_competencies = []
    mandatory_tokens = ["must", "required", "mandatory", "essential", "minimum", "proficiency in", "strong knowledge"]
    
    for comp_name, comp_info in TAXONOMY.items():
        matched_aliases = []
        for alias in comp_info["aliases"]:
            if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", norm):
                matched_aliases.append(alias)
        
        if matched_aliases:
            evidence = []
            for s in sentences:
                if any(alias in s.lower() for alias in matched_aliases):
                    evidence.append(s.strip())
                    if len(evidence) >= 2:
                        break
            
            is_mandatory = any(
                any(tok in s.lower() for tok in mandatory_tokens)
                for s in evidence
            ) or (comp_name in ["Problem Solving & DSA", "REST APIs & Web Services"])

            confidence = min(0.98, 0.65 + 0.08 * len(matched_aliases))
            
            prof = "Intermediate"
            if seniority in ["Senior", "Lead"]:
                prof = "Advanced"
            elif seniority in ["Entry", "Junior"]:
                prof = "Beginner"

            found_competencies.append({
                "competency": comp_name,
                "category": comp_info["category"],
                "mandatory": is_mandatory,
                "proficiency": prof,
                "confidence": round(confidence, 2),
                "matched_terms": sorted(set(matched_aliases)),
                "evidence": evidence,
                "importance": 5 if is_mandatory else 3,
            })

    tools = [c["competency"] for c in found_competencies if c["category"] == "Tools"]
    behaviors = [c["competency"] for c in found_competencies if c["category"] == "Behavioral"]
    if not behaviors:
        behaviors = ["Stakeholder Communication", "Agile & Ownership"]

    ambiguities = []
    if len(found_competencies) < 3:
        ambiguities.append("Few skills were explicitly detected. Review JD text or expand taxonomy.")
    if not any(c["mandatory"] for c in found_competencies):
        ambiguities.append("No explicit mandatory skills marked in text ('must', 'required'). Please confirm core requirements.")

    return {
        "role_title": f"{seniority} Software Engineer",
        "seniority": seniority,
        "responsibilities": responsibilities[:5],
        "competencies": found_competencies,
        "tools": tools,
        "observable_behaviors": behaviors,
        "ambiguities": ambiguities,
    }


def test_openai_api_key(api_key: str, model: str = "gpt-4o-mini") -> tuple[bool, str]:
    if not api_key or not api_key.strip():
        return False, "API key is empty."
    if requests is None:
        return False, "The 'requests' module is missing."

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=10)
        if resp.status_code == 200:
            return True, f"Authentication succeeded! Model catalog reachable. Selected model: {model}"
        elif resp.status_code == 401:
            return False, "401 Unauthorized: The provided API key is invalid or revoked."
        elif resp.status_code == 429:
            return False, "429 Rate Limit / Quota Exceeded: Your account has hit a rate limit or has insufficient credits."
        else:
            return False, f"API test returned HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as exc:
        return False, f"Connection failed: {str(exc)}"


def call_openai_chat(prompt: str, api_key: str, model: str = "gpt-4o-mini", system_role: str = "You are an expert technical recruiter and assessment architect.") -> str:
    if requests is None:
        raise RuntimeError("Install requests: pip install requests")

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=90)
    if not resp.ok:
        # Fallback to responses endpoint if configured
        resp_fallback = requests.post("https://api.openai.com/v1/responses", headers=headers, json={"model": model, "input": prompt}, timeout=90)
        if resp_fallback.ok:
            data = resp_fallback.json()
            return data.get("output_text", "")
        raise RuntimeError(f"OpenAI API Error {resp.status_code}: {resp.text[:400]}")

    data = resp.json()
    return data["choices"][0]["message"]["content"]


def ai_based_extraction(jd_text: str, api_key: str, model: str) -> dict:
    prompt = f"""
Analyze this Job Description and return ONLY valid JSON matching this schema:
{{
  "role_title": "string",
  "seniority": "Entry|Junior|Mid|Senior|Lead",
  "responsibilities": ["2 to 5 primary responsibilities extracted verbatim or summarized"],
  "tools": ["list of explicit tools, e.g. Docker, Git, AWS, Jira"],
  "observable_behaviors": ["list of expected behaviors, e.g. Code reviews, Mentoring, Root cause debugging"],
  "ambiguities": ["any vague or ambiguous requirements in the JD that need human review"],
  "competencies": [
    {{
      "name": "Canonical skill name aligned with standard tech roles",
      "category": "Technical|Tools|Behavioral",
      "mandatory": true,
      "proficiency": "Beginner|Intermediate|Advanced",
      "importance": 1 to 5,
      "confidence": 0.70 to 0.99,
      "evidence": ["short exact sentence or phrase from the JD"]
    }}
  ]
}}

Job Description:
{jd_text[:14000]}
"""
    raw_response = call_openai_chat(prompt, api_key, model)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_response.strip(), flags=re.I)
    parsed = json.loads(cleaned)

    normalized = []
    for item in parsed.get("competencies", []):
        raw_name = item.get("name", "General Skill")
        matched_canon = raw_name
        for canon, data in TAXONOMY.items():
            if canon.lower() in raw_name.lower() or any(alias in raw_name.lower() for alias in data["aliases"]):
                matched_canon = canon
                break
        
        normalized.append({
            "competency": matched_canon,
            "category": item.get("category", "Technical"),
            "mandatory": bool(item.get("mandatory", False)),
            "proficiency": item.get("proficiency", "Intermediate"),
            "importance": int(item.get("importance", 3)),
            "confidence": float(item.get("confidence", 0.85)),
            "evidence": item.get("evidence", []),
            "matched_terms": [raw_name],
        })

    deduped = {}
    for c in normalized:
        name = c["competency"]
        if name not in deduped or c["confidence"] > deduped[name]["confidence"]:
            deduped[name] = c

    parsed["competencies"] = list(deduped.values())
    return parsed


def get_adaptive_difficulty_mix(seniority: str) -> dict:
    if seniority in ["Entry", "Junior"]:
        return {"Easy": 50, "Medium": 40, "Hard": 10}
    elif seniority == "Mid":
        return {"Easy": 25, "Medium": 55, "Hard": 20}
    elif seniority in ["Senior", "Lead"]:
        return {"Easy": 10, "Medium": 45, "Hard": 45}
    return {"Easy": 30, "Medium": 50, "Hard": 20}


def calculate_jaccard_similarity(str1: str, str2: str) -> float:
    words1 = set(re.findall(r"\w+", str1.lower()))
    words2 = set(re.findall(r"\w+", str2.lower()))
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)


def build_adaptive_blueprint(competencies: list, seniority: str, total_questions: int, duration: int, easy_pct: int, medium_pct: int, hard_pct: int) -> tuple[list, list]:
    if not competencies:
        return [], ["No competencies provided. Extract or add skills first."]
    
    if easy_pct + medium_pct + hard_pct != 100:
        return [], [f"Difficulty percentages sum to {easy_pct + medium_pct + hard_pct}%, but must equal 100%."]

    for comp in competencies:
        base = 4 if comp.get("mandatory") else 2
        conf_boost = round(float(comp.get("confidence", 0.6)) * 2)
        comp["calculated_weight"] = max(1, base + conf_boost)

    total_weight = sum(c["calculated_weight"] for c in competencies)
    for comp in competencies:
        comp["weight_pct"] = round((comp["calculated_weight"] / total_weight) * 100, 1)

    raw_alloc = [(total_questions * c["calculated_weight"]) / total_weight for c in competencies]
    counts = [int(x) for x in raw_alloc]
    remainder = total_questions - sum(counts)
    for idx in sorted(range(len(raw_alloc)), key=lambda i: raw_alloc[i] - counts[i], reverse=True)[:remainder]:
        counts[idx] += 1

    blueprint = []
    warnings = []
    used_questions = []

    for comp, allocated_count in zip(competencies, counts):
        comp_name = comp["competency"]
        candidates = [q for q in QUESTION_BANK if q["competency"] == comp_name]

        desired_difficulties = []
        for diff_name, pct in [("Easy", easy_pct), ("Medium", medium_pct), ("Hard", hard_pct)]:
            desired_difficulties.extend([diff_name] * int(round(allocated_count * pct / 100)))
        while len(desired_difficulties) < allocated_count:
            desired_difficulties.append("Medium")
        desired_difficulties = desired_difficulties[:allocated_count]

        selected_for_comp = []
        for diff in desired_difficulties:
            matching = [
                q for q in candidates 
                if q["difficulty"] == diff and q["id"] not in [u["id"] for u in used_questions]
            ]
            if not matching:
                matching = [q for q in candidates if q["id"] not in [u["id"] for u in used_questions]]

            if matching:
                chosen = dict(matching[0])
                for existing in used_questions:
                    sim = calculate_jaccard_similarity(chosen["question"], existing["question"])
                    if sim > 0.45:
                        warnings.append(f"Overlap Guardrail: Question '{chosen['id']}' shares {int(sim*100)}% keyword overlap with '{existing['id']}'.")
                        break

                used_questions.append(chosen)
                selected_for_comp.append(chosen)

        if len(selected_for_comp) < allocated_count:
            warnings.append(f"Coverage Gap: Competency '{comp_name}' needs {allocated_count} questions, but only {len(selected_for_comp)} found in bank.")

        blueprint.append({
            "competency": comp_name,
            "category": comp.get("category", "Technical"),
            "mandatory": comp.get("mandatory", False),
            "proficiency": comp.get("proficiency", "Intermediate"),
            "weight_pct": comp["weight_pct"],
            "allocated_questions": allocated_count,
            "selected_questions": selected_for_comp,
            "evidence": comp.get("evidence", []),
            "confidence": comp.get("confidence", 0.7),
        })

    total_est_minutes = sum(q["minutes"] for b in blueprint for q in b["selected_questions"])
    if total_est_minutes > duration:
        warnings.append(f"Duration Warning: Total question time is {total_est_minutes} min, exceeding target of {duration} min.")
    elif total_est_minutes < duration * 0.60:
        warnings.append(f"Duration Warning: Total question time is {total_est_minutes} min, significantly below target of {duration} min.")

    return blueprint, warnings


def generate_gap_question(competency: str, jd_text: str, difficulty: str, api_key: str, model: str = "gpt-4o-mini") -> dict:
    prompt = f"""
Create one high-quality, scenario-based or work-sample assessment question for:
Competency: {competency}
Target Difficulty: {difficulty}
Role Context: Use the job description below to ground the scenario in actual company tasks.

Return ONLY a valid JSON object matching this schema:
{{
  "id": "AI-GEN-{random.randint(100, 999)}",
  "competency": "{competency}",
  "subskill": "specific subtopic",
  "difficulty": "{difficulty}",
  "cognitive_level": "Apply/Analyze or Evaluate/Synthesize",
  "job_relevance": "Direct Core Requirement",
  "type": "Scenario",
  "question": "Detailed scenario question presenting an engineering problem",
  "expected": "Model answer and key architectural / algorithmic trade-offs expected",
  "rubric": "Grading scale: 3 pts for optimal solution, 2 for working brute force, 0 for failure",
  "minutes": 8,
  "review_status": "Draft — human review required before publication"
}}

Job Description:
{jd_text[:8000]}
"""
    raw = call_openai_chat(prompt, api_key, model)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    data = json.loads(cleaned)
    data["review_status"] = "Draft — human review required"
    return data


def run_candidate_simulation(blueprint: list, duration: int = 60) -> dict | None:
    all_questions = [q for group in blueprint for q in group["selected_questions"]]
    if not all_questions:
        return None

    profiles = {
        "Candidate A: Senior Expert": {
            "description": "Strong fundamentals, deep distributed design & edge case handling",
            "time_factor": 0.82,
            "success_prob": {"Easy": 0.98, "Medium": 0.90, "Hard": 0.82},
        },
        "Candidate B: Mid Competent": {
            "description": "Solid core skills, partial answers on complex architecture",
            "time_factor": 1.05,
            "success_prob": {"Easy": 0.90, "Medium": 0.72, "Hard": 0.40},
        },
        "Candidate C: Junior / Unprepared": {
            "description": "Solves basic syntax, struggles with scenarios and timing",
            "time_factor": 1.35,
            "success_prob": {"Easy": 0.70, "Medium": 0.35, "Hard": 0.10},
        },
    }

    results = []
    for name, p in profiles.items():
        score = 0
        total_possible = 0
        sim_time = 0

        for q in all_questions:
            diff = q.get("difficulty", "Medium")
            prob = p["success_prob"].get(diff, 0.6)
            q_time = q.get("minutes", 5) * p["time_factor"] * (0.95 + 0.1 * random.random())
            sim_time += q_time
            
            pts = 10 if diff == "Easy" else (20 if diff == "Medium" else 30)
            earned = pts if random.random() < prob else (pts * 0.4 if random.random() < prob else 0)
            score += earned
            total_possible += pts

        final_pct = round((score / max(1, total_possible)) * 100, 1)
        sim_minutes = round(sim_time, 1)
        
        status = "Within Time"
        if sim_minutes > duration:
            status = f"Exceeded Limit (+{round(sim_minutes - duration, 1)}m)"

        results.append({
            "Profile": name,
            "Description": p["description"],
            "Simulated Score (%)": final_pct,
            "Simulated Duration (min)": sim_minutes,
            "Time Status": status,
        })

    scores = [r["Simulated Score (%)"] for r in results]
    separation_index = round(max(scores) - min(scores), 1)

    return {
        "results": results,
        "separation_index": separation_index,
        "is_discriminating": separation_index >= 35.0,
    }


def highlight_jd_text(jd_text: str, competencies: list) -> str:
    if not jd_text or not competencies:
        return jd_text

    highlighted = jd_text
    palette = ["#fef08a", "#bbf7d0", "#bae6fd", "#fed7aa", "#e9d5ff", "#fecdd3"]
    
    term_color_map = {}
    for i, comp in enumerate(competencies):
        color = palette[i % len(palette)]
        for term in comp.get("matched_terms", []):
            if len(term) >= 3:
                term_color_map[term] = (comp["competency"], color)

    sorted_terms = sorted(term_color_map.keys(), key=len, reverse=True)
    for term in sorted_terms:
        comp_name, color = term_color_map[term]
        pattern = re.compile(r"(?i)\b(" + re.escape(term) + r")\b")
        replacement = f'<mark style="background-color: {color}; padding: 2px 5px; border-radius: 4px; font-weight: 600;" title="{comp_name} ({term})">\\1</mark>'
        highlighted = pattern.sub(replacement, highlighted)

    return highlighted
