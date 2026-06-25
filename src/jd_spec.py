"""
jd_spec.py
==========
A *structured*, machine-readable encoding of the released Job Description
("Senior AI Engineer - Founding Team", Redrob AI).

The whole point of this challenge is that you cannot rank well by keyword
matching. So instead of treating the JD as a bag of words, we read it the way
a recruiter does and distil it into:

  1. What the role genuinely needs   -> CORE_RETRIEVAL_SKILLS, EVIDENCE_PHRASES
  2. What is "nice to have"           -> NICE_TO_HAVE_SKILLS
  3. What the JD explicitly rejects   -> DISTRACTOR_SKILLS, SERVICES_COMPANIES,
                                         NON_TECH_TITLE_FAMILIES, title-chasing
  4. Where the role sits              -> TARGET_CITIES, experience band
  5. An "ideal candidate" paragraph   -> IDEAL_PROFILE_TEXT (used for semantic
                                         similarity so we catch *plain-language*
                                         strong candidates who never say "RAG")

Every weight and list here is a design decision we can defend in the Stage-5
interview; nothing is hidden inside the scorer.
"""

from __future__ import annotations

# Experience band: ideal is 6-8 years, but 5-9 is still acceptable.
EXP_IDEAL_LOW = 6.0
EXP_IDEAL_HIGH = 8.0
EXP_SOFT_LOW = 5.0
EXP_SOFT_HIGH = 9.0

# Skills that matter most for this Senior AI Engineer role.
CORE_RETRIEVAL_SKILLS = {
    "embeddings", "vector search", "semantic search", "sentence transformers",
    "faiss", "pinecone", "qdrant", "weaviate", "milvus", "pgvector",
    "information retrieval", "information retrieval systems",
    "recommendation systems", "ranking systems", "vector representations",
    "rag", "llamaindex", "hugging face transformers",
}

# Useful ML/LLM skills, slightly below retrieval/ranking skills.
CORE_ML_SKILLS = {
    "llms", "pytorch", "tensorflow", "fine-tuning llms", "prompt engineering",
    "nlp", "machine learning", "deep learning",
}

# Nice-to-have skills.
NICE_TO_HAVE_SKILLS = {
    "lora", "qlora", "peft",            # LLM fine-tuning
    "xgboost", "learning to rank",      # learning-to-rank
    "elasticsearch", "opensearch", "bm25",
}

# MLOps/infra skills: useful, but not the main requirement.
MLOPS_SKILLS = {
    "kubernetes", "docker", "mlflow", "kubeflow", "bentoml", "databricks",
    "airflow", "spark", "kafka", "ci/cd", "terraform", "aws", "gcp",
    "ray", "triton",
}

# Distractor skills: AI-adjacent, but not the core NLP/IR focus of this job.
DISTRACTOR_SKILLS = {
    "computer vision", "image classification", "object detection",
    "speech recognition", "tts", "opencv", "ocr", "image segmentation",
}

# Career evidence phrases found in summaries and job descriptions.
# Higher weight means stronger evidence for this role.
EVIDENCE_PHRASES = {
    # Retrieval/search/ranking/recommendation: the heart of the role.
    "recommendation system": 1.0, "recommender": 1.0, "recommendation engine": 1.0,
    "search system": 1.0, "search engine": 0.9, "search relevance": 1.0,
    "ranking system": 1.0, "ranking model": 1.0, "learning to rank": 1.0,
    "learning-to-rank": 1.0, "re-ranking": 1.0, "reranking": 1.0,
    "information retrieval": 1.0, "retrieval": 0.8, "semantic search": 1.0,
    "vector search": 1.0, "vector database": 1.0, "nearest neighbor": 0.8,
    "personalization": 0.8, "personalisation": 0.8, "relevance": 0.6,
    "matching": 0.5, "candidate matching": 0.9,
    "embedding": 0.9, "embeddings": 0.9, "bm25": 0.8, "hybrid search": 1.0,
    "elasticsearch": 0.7, "opensearch": 0.7, "solr": 0.6, "faiss": 0.9,
    "hnsw": 1.0, "colbert": 1.0, "cross-encoder": 0.9, "bi-encoder": 0.9,
    "dense retrieval": 1.0, "sparse retrieval": 0.9, "late interaction": 0.9,
    # LLM/NLP production.
    "rag": 0.9, "retrieval-augmented": 1.0, "fine-tun": 0.7, "fine tuned": 0.7,
    "transformer": 0.6, "language model": 0.7, "llm": 0.7, "nlp": 0.6,
    # Evaluation quality.
    "ndcg": 1.0, "mrr": 0.9, "map@": 0.7, "mean average precision": 0.9,
    "a/b test": 0.8, "ab test": 0.8, "offline evaluation": 0.9,
    "evaluation framework": 0.9, "offline-to-online": 1.0,
    # Production and scale.
    "production": 0.5, "deployed": 0.5, "real users": 0.7, "at scale": 0.6,
    "low latency": 0.5, "serving": 0.4, "inference": 0.4, "shipped": 0.5,
}

# Phrases that suggest the candidate may have stopped hands-on coding.
NON_CODING_PHRASES = {
    "managed a team", "led a team of", "people management", "stakeholder",
    "roadmap", "headcount", "org design", "purely architectural",
    "no longer hands-on", "stepped back from coding",
}

# Title relevance. Value is 0..1, where 1 is strongest.
TITLE_RELEVANCE = {
    "machine learning engineer": 1.0, "ml engineer": 1.0,
    "applied scientist": 1.0, "ai engineer": 0.95, "nlp engineer": 1.0,
    "research scientist": 0.75,            # research alone is risky (see disqualifier)
    "ai research": 0.8, "ai specialist": 0.85, "data scientist": 0.8,
    "search engineer": 1.0, "relevance engineer": 1.0,
    "recommendation": 1.0, "personalization": 0.95,
    "staff engineer": 0.8, "principal engineer": 0.8,
    "senior software engineer": 0.7, "software engineer": 0.6,
    "backend engineer": 0.65, "platform engineer": 0.6,
    "data engineer": 0.6, "analytics engineer": 0.5, "mlops": 0.7,
    "deep learning": 0.85,
}

# Non-technical title families. These help catch keyword-stuffers.
NON_TECH_TITLE_FAMILIES = {
    "hr manager", "human resources", "recruiter", "talent acquisition",
    "sales executive", "sales manager", "account executive",
    "marketing manager", "marketing executive", "seo specialist",
    "accountant", "accounts", "finance manager", "auditor",
    "content writer", "copywriter", "content strategist",
    "graphic designer", "ui designer", "visual designer", "illustrator",
    "customer support", "customer success", "support engineer",
    "operations manager", "operations executive", "supply chain",
    "project manager", "program manager", "scrum master",
    "business analyst", "business development",
    "mechanical engineer", "civil engineer", "electrical engineer",
    "production engineer", "quality engineer",
}

# Services companies. Services-only careers get a penalty.
SERVICES_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "mindtree", "mphasis", "hexaware",
    "ltimindtree", "l&t infotech", "lti", "igate", "syntel", "birlasoft",
    "persistent systems", "ust global", "ust", "nttdata", "ntt data",
}

# Product companies. Experience here gets a small positive signal.
PRODUCT_COMPANY_HINTS = {
    "google", "meta", "facebook", "amazon", "microsoft", "apple", "netflix",
    "flipkart", "swiggy", "zomato", "uber", "ola", "razorpay", "cred",
    "phonepe", "paytm", "myntra", "meesho", "sharechat", "dream11",
    "navi", "groww", "zerodha", "freshworks", "zoho", "postman", "browserstack",
    "hasura", "atlassian", "linkedin", "twitter", "pinterest", "spotify",
    "databricks", "nvidia", "openai", "anthropic", "cohere", "huggingface",
    "hugging face", "scale ai", "adobe", "salesforce", "uber", "airbnb",
    "walmart", "target", "intuit", "stripe", "doordash", "instacart",
}

# Location preferences.
PREFERRED_CITIES = {"pune", "noida"}
WELCOME_CITIES = {
    "hyderabad", "mumbai", "delhi", "new delhi", "gurgaon", "gurugram",
    "ghaziabad", "faridabad", "greater noida",  # Delhi NCR
}
TIER1_CITIES = {
    "bangalore", "bengaluru", "chennai", "kolkata", "ahmedabad", "jaipur",
}

# Ideal candidate text used by semantic.py for meaning-based matching.
IDEAL_PROFILE_TEXT = (
    "Senior applied machine learning engineer with about six to eight years of "
    "experience, four to five of them building production machine learning at "
    "product companies rather than services firms. Has shipped an end-to-end "
    "ranking, search, retrieval, recommendation or matching system to real "
    "users at meaningful scale. Deep with embeddings-based retrieval using "
    "sentence transformers and vector databases such as FAISS, Pinecone, "
    "Qdrant, Weaviate or Milvus, and with hybrid search combining lexical and "
    "dense retrieval. Strong Python engineer who still writes production code. "
    "Designs rigorous evaluation frameworks for ranking systems using NDCG, "
    "MRR, MAP and offline-to-online A/B testing. Comfortable with LLMs, "
    "transformers and fine-tuning. Scrappy product-minded shipper who learns "
    "from real users. Based in or willing to relocate to Pune, Noida, "
    "Hyderabad, Mumbai or Delhi NCR, active in the job market and responsive "
    "to recruiters."
)

# Component weights for base_fit. These add up to 1.0.
WEIGHTS = {
    "title_career": 0.26,   # decisive signal vs keyword stuffers
    "career_evidence": 0.22,  # rescues plain-language Tier-5s
    "skills_trust": 0.18,   # trust-weighted JD-core skills
    "semantic": 0.14,       # fuzzy JD similarity
    "experience": 0.10,
    "location": 0.06,
    "education": 0.04,
}
