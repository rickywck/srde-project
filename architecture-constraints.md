# Architecture Constraints: Agentic AI-Powered Credit Card Dispute Resolution System

This document extracts architectural constraints from the system design to guide implementation decisions and ensure compliance with design principles.

---

## 1. Platform & Technology Constraints

### 1.1 Cloud Platform
- **MUST** use Google Cloud Platform (GCP) as the primary cloud platform
- **MUST** leverage GCP-native services for core functionality
- **MUST** implement cloud-native design patterns for scalability

### 1.2 AI/ML Stack
- **MUST** deploy all agents (orchestrator, triage, analysis) on Vertex AI Agent Engine
- **MUST** use Google ADK (Agent Developer Kit) for agent development
- **MUST** use Vertex AI-hosted LLM models (Gemini Pro, Gemini Ultra) for reasoning and NLP tasks
- **MUST** use Vertex AI Embeddings (text-embedding-gecko or text-embedding-004) for vector generation
- **MUST** implement agent reasoning with explainable outputs (reasoning chains, confidence scores)

### 1.3 Compute & Hosting
- **MUST** deploy customer channels (web portal, mobile backend) on Cloud Run
- **MUST** deploy agent tools/connectors on Cloud Run as serverless containers
- **MUST** configure auto-scaling for all Cloud Run services based on traffic patterns
- **MUST** use Dialogflow CX for chatbot integration with Vertex AI agents
- **SHOULD** use Cloud Run jobs for background processing (document processing, embedding generation)

### 1.4 Programming Languages
- **MUST** use Python 3.11+ for agent code with Google ADK
- **SHOULD** use Go or Java/Kotlin for performance-critical connectors and tools
- **SHOULD** use React or Vue.js with TypeScript for web frontend
- **SHOULD** use Swift (iOS) and Kotlin (Android) for native mobile apps, or React Native for cross-platform

---

## 2. Data Architecture Constraints

### 2.1 Document Management
- **MUST** store all raw documents in Cloud Storage with customer-managed encryption keys (CMEK)
- **MUST** organize documents by dispute_id with versioning enabled
- **MUST** implement bucket lifecycle policies for long-term archival
- **MUST** maintain audit trails for all document access

### 2.2 Knowledge Management (Vector Store)
- **MUST** implement semantic search using Vertex AI Vector Search or Weaviate
- **MUST** index both static knowledge (policies, FAQs, regulations) and dynamic knowledge (case documents, past resolutions)
- **MUST** support real-time indexing of new case documents (within seconds)
- **MUST** use Tree-AH index type for low-latency semantic search
- **MUST** generate embeddings from raw documents via Vertex AI Embeddings
- **SHOULD** update static knowledge daily via batch processing

### 2.3 Operational Database
- **MUST** use Cloud SQL (PostgreSQL) with high availability for relational dispute data, OR Firestore for flexible document model
- **MUST** support real-time sync capabilities for status updates
- **MUST** implement proper indexing for query performance

### 2.4 Data Lake & Analytics
- **MUST** use BigQuery for analytics, ML training datasets, and historical archives
- **MUST** partition data by date for query performance
- **MUST** implement streaming inserts via Cloud Pub/Sub for completed cases
- **MUST NOT** use BigQuery for real-time dispute processing (use Core Banking directly instead)
- **MUST** store completed transaction history, AI predictions vs. actuals, human feedback, and representment results

### 2.5 Data Flow Constraints
- **MUST** follow this flow for document ingestion: Raw documents → Cloud Storage (DMS) → Embedding generation → Vector Store indexing → Agent queries
- **MUST** follow this flow for completed cases: Orchestrator → Pub/Sub → BigQuery → MLOps pipelines
- **MUST** load historical transaction data from Core Banking to BigQuery via periodic batch processes

---

## 3. Agent Architecture Constraints

### 3.1 Agent Deployment
- **MUST** implement Orchestrator Agent as the single entry point for all workflows
- **MUST** separate Triage Agents from Analysis Agents for clear separation of concerns
- **MUST** deploy all agents on Vertex AI Agent Engine (not custom infrastructure)
- **MUST** use Google ADK for agent tool integration

### 3.2 Triage Agent Requirements
- **MUST** perform schema validation before business logic processing
- **MUST** check for duplicate disputes on the same transaction
- **MUST** apply policy gating rules (transaction date windows, dispute type eligibility)
- **MUST** assess evidence completeness
- **MUST** generate confidence scores (0-100) for all cases
- **MUST** output Valid/Invalid/RequireHuman status with detailed reasoning
- **MUST** route cases based on confidence thresholds:
  - <60: Mandatory human review
  - 60-80: Parallel human oversight flagged
  - >80: Proceed to automated analysis

### 3.3 Analysis Agent Requirements
- **MUST** query Core Banking for real-time transaction details via tools
- **MUST** query CRM for customer profile and history via tools
- **MUST** perform semantic search against Knowledge Management for policies and similar cases
- **MUST** retrieve raw documents from DMS when detailed review is needed
- **MUST** generate structured reasoning chains for all decisions
- **MUST** output recommended actions (Approve/Reject/RequestInfo/Escalate) with confidence scores
- **MUST** produce evidence summaries suitable for representment packages

### 3.4 Orchestrator Requirements
- **MUST** coordinate all workflow stages: triage → retrieval → analysis → tagging → resolution
- **MUST** manage workflow state and handle escalations based on confidence scores
- **MUST** coordinate provisional credit decisions and chargeback submissions
- **MUST** route to human workbenches when confidence thresholds are not met
- **MUST** implement timeout and retry logic for agent and tool calls
- **MUST** log all decisions to AgentEval for model improvement

### 3.5 Agent Tools/Connectors
- **MUST** implement all backend integrations as separate tool services (not direct agent access)
- **MUST** use least-privilege service accounts with workload identity
- **MUST** store all secrets in Secret Manager
- **MUST** implement circuit breakers and retry logic for all external system calls
- **MUST** log all tool invocations for compliance and debugging
- **MUST** provide auditable, read-only access patterns to Core Banking

---

## 4. Security & Compliance Constraints

### 4.1 Encryption
- **MUST** encrypt all data at rest using customer-managed encryption keys (CMEK) via Cloud KMS
- **MUST** encrypt all data in transit using TLS 1.3
- **MUST** never store unencrypted PII in logs or non-production environments

### 4.2 Access Control
- **MUST** implement Role-Based Access Control (RBAC) for all system components
- **MUST** follow least-privilege principle for all service accounts and IAM roles
- **MUST** use Workload Identity for service-to-service authentication
- **MUST** use Identity Platform (Firebase Auth) for customer-facing authentication
- **MUST** implement OAuth2/OIDC for API authentication

### 4.3 Audit & Compliance
- **MUST** enable Cloud Audit Logs for all data access and administrative actions
- **MUST** maintain audit trails for all document access
- **MUST** log all agent decisions with timestamps and reasoning
- **MUST** track all model versions, training data, and decisions for regulatory compliance
- **MUST** implement PII masking in logs and non-production environments
- **MUST** support data residency controls for regional compliance

### 4.4 Data Privacy
- **MUST** anonymize/mask sensitive data for AI model training where appropriate
- **MUST** implement data retention and deletion policies per regulatory requirements
- **MUST** provide customer data access and deletion capabilities (GDPR, CCPA compliance)

---

## 5. Integration & API Constraints

### 5.1 API Gateway
- **MUST** route all channel requests through a managed API Gateway (Cloud Endpoints or Apigee)
- **MUST** implement authentication (OAuth2/OIDC) at the gateway level
- **MUST** implement rate limiting to prevent abuse
- **MUST** perform protocol-level validation (schema validation) at ingress
- **MUST NOT** perform business validation at the gateway (delegate to Triage Agents)

### 5.2 Integration Patterns
- **MUST** use API-first approach for all service interfaces
- **MUST** use event-driven architecture via Cloud Pub/Sub for asynchronous communication
- **MUST** use standard protocols (REST, gRPC, OAuth2)
- **MUST** implement circuit breakers for all external system integrations
- **MUST** use read-only access patterns for Core Banking queries

### 5.3 External System Integration
- **MUST** integrate with Core Banking for real-time transaction and account data
- **MUST** integrate with CRM for customer profiles and interaction history
- **MUST** integrate with acquirer/card network APIs for chargeback submissions
- **MUST** integrate with merchant evidence endpoints for representment
- **MUST** query fraud detection systems via tools/connectors
- **MUST** implement retry logic and timeout handling for all external calls

---

## 6. User Experience Constraints

### 6.1 Customer Interface
- **MUST** support 3-minute dispute submission experience
- **MUST** provide multi-channel access (web portal, mobile app, chatbot)
- **MUST** implement biometric authentication for mobile apps
- **MUST** support real-time status updates via push notifications, SMS, or email
- **MUST** provide self-service portal for status tracking and evidence submission
- **MUST** implement progressive disclosure and AI-guided intake forms
- **MUST** support document upload with OCR preview
- **MUST** provide resumable upload capability for large documents
- **SHOULD** implement Progressive Web App (PWA) for mobile-first experience

### 6.2 Agent Interface
- **MUST** display AI recommendations with confidence scores (0-100) and reasoning chains
- **MUST** provide unified case workbench with all information in a single view
- **MUST** implement intelligent case queue prioritization
- **MUST** support one-click actions for common decisions (Approve/Reject/RequestInfo/Escalate)
- **MUST** capture all human decisions with timestamps and override reasons
- **MUST** provide integrated search to Knowledge Management for policy verification
- **MUST** color-code confidence levels (green >80, yellow 60-80, red <60)
- **SHOULD** implement keyboard shortcuts for power users
- **SHOULD** use lazy loading and virtualization for large evidence sets

---

## 7. AI/ML Lifecycle Constraints (MLOps)

### 7.1 Model Versioning & Registry
- **MUST** track all model versions in Vertex AI Model Registry
- **MUST** tag each agent version with git commit SHA and performance baseline metrics
- **MUST** maintain lineage between models, training data, and deployed agents

### 7.2 Training & Retraining
- **MUST** implement monthly scheduled batch retraining on accumulated historical data
- **MUST** implement trigger-based retraining when accuracy drops below threshold or drift is detected
- **MUST** use Vertex Pipelines to orchestrate data preparation, training, evaluation, and deployment
- **MUST** source training data from BigQuery (completed cases, human feedback, outcomes)
- **MUST** validate model performance before promoting to production

### 7.3 Monitoring & Observability
- **MUST** monitor prediction distribution shifts and feature drift
- **MUST** track accuracy, precision, recall, and confidence calibration against live outcomes
- **MUST** implement automated alerts when metrics degrade beyond acceptable thresholds
- **MUST** provide real-time monitoring dashboard for ML team visibility
- **MUST** log all AI predictions, confidence scores, and human decisions via AgentEval Service

### 7.4 Explainability (XAI)
- **MUST** produce step-by-step reasoning chains for all LLM agent decisions
- **MUST** highlight key evidence and factors influencing recommendations
- **MUST** surface past similar cases that informed current decision
- **MUST** explain confidence score components (evidence quality, policy alignment, pattern match)

### 7.5 Feedback Loop
- **MUST** capture all human overrides with rationale in real-time
- **MUST** analyze override patterns weekly to identify systematic issues
- **MUST** link final case outcomes (representment results, customer satisfaction) back to original AI predictions
- **MUST** store feedback in BigQuery with structured metadata for MLOps consumption
- **MUST** support A/B testing for new agent versions on subset of traffic
- **SHOULD** implement dynamic tuning of confidence thresholds based on accuracy/efficiency trade-offs

---

## 8. Performance & Scalability Constraints

### 8.1 Response Time Requirements
- **MUST** return triage decision within 5 seconds of submission
- **MUST** return analysis recommendation within 30 seconds for automated cases
- **SHOULD** complete end-to-end processing (triage → analysis → decision) within 2 minutes for high-confidence cases

### 8.2 Scalability Requirements
- **MUST** support elastic scaling via Cloud Run auto-scaling
- **MUST** handle peak submission volumes without degradation
- **MUST** use asynchronous processing via Cloud Pub/Sub for high-volume events
- **MUST** implement proper indexing and partitioning for database queries
- **MUST** use connection pooling for database connections

### 8.3 Availability & Resilience
- **MUST** deploy across multiple availability zones for high availability
- **MUST** implement robust backup and disaster recovery strategies
- **MUST** use circuit breakers to handle transient failures in integrated systems
- **MUST** implement graceful degradation when external systems are unavailable
- **SHOULD** achieve 99.9% uptime for customer-facing services

---

## 9. Observability & Monitoring Constraints

### 9.1 Logging
- **MUST** implement centralized logging via Cloud Logging
- **MUST** use structured JSON logs for all services
- **MUST** implement log-based metrics for key business events
- **MUST** mask PII in all logs

### 9.2 Tracing
- **MUST** implement distributed tracing via Cloud Trace
- **MUST** track requests across all microservices and agents
- **MUST** identify performance bottlenecks via trace analysis

### 9.3 Metrics & Alerting
- **MUST** collect and monitor these KPIs:
  - Agent performance: latency, confidence scores, accuracy
  - Case processing: resolution time, escalation rate, throughput
  - Business metrics: customer satisfaction, operational cost per case
- **MUST** implement dashboards via Cloud Monitoring
- **MUST** configure alerts for critical failures and degraded performance
- **MUST** use Error Reporting for exception aggregation

---

## 10. Development & Deployment Constraints

### 10.1 Infrastructure as Code
- **MUST** use Terraform for all infrastructure provisioning and management
- **MUST** maintain all infrastructure definitions in version control
- **MUST** implement drift detection and remediation

### 10.2 CI/CD Pipeline
- **MUST** use Cloud Build or GitHub Actions for automated build and test
- **MUST** store container images in Artifact Registry
- **MUST** use Cloud Deploy for progressive delivery (canary, blue-green deployments)
- **MUST** run unit tests in CI pipeline before merge
- **MUST** run integration tests against staging environment before production deployment
- **SHOULD** implement automated load testing with Cloud Load Testing or k6

### 10.3 Environment Management
- **MUST** maintain separate Dev, Staging, and Production environments
- **MUST** isolate environments with separate GCP projects and VPCs
- **MUST** use consistent configuration management across environments
- **MUST** test all changes in staging before production deployment

---

## 11. Workflow & Business Logic Constraints

### 11.1 Dispute Processing Flow
- **MUST** follow this sequence: Ingestion → Triage → Analysis → Decision → Resolution
- **MUST** validate and filter at triage stage before expensive analysis
- **MUST** route low-confidence cases (<60) to human review before analysis
- **MUST** flag medium-confidence cases (60-80) for parallel human oversight
- **MUST** allow automated resolution only for high-confidence cases (>90)
- **MUST** store all case data and decisions for future training

### 11.2 Human-in-the-Loop Requirements
- **MUST** present AI recommendations to humans with confidence scores and reasoning
- **MUST** allow humans to accept, override, or request more context
- **MUST** capture human decisions with timestamps, agent ID, and override reasoning
- **MUST** immediately feed human corrections back to AgentEval for model improvement
- **MUST** label feedback as "AI_CORRECT" or override with rationale

### 11.3 Chargeback Workflow
- **MUST** coordinate provisional credit issuance via Orchestrator and Core Banking
- **MUST** generate evidence packages suitable for acquirer submission
- **MUST** track chargeback status and representment results
- **MUST** reconcile provisional credit based on final outcome (capture or reversal)
- **MUST** store chargeback outcomes in BigQuery for model training

---

## 12. Communication & Notification Constraints

### 12.1 Customer Communication
- **MUST** support multi-channel notifications: email, SMS, push notifications
- **MUST** send immediate confirmation upon dispute submission
- **MUST** provide real-time status updates during processing
- **MUST** use AI-generated personalized messages with customer context
- **MUST** integrate with CRM for communication history tracking
- **SHOULD** use SendGrid/Cloud Email for email, Twilio for SMS, Firebase Cloud Messaging for push

### 12.2 Internal Communication
- **MUST** notify human agents of assigned cases in real-time
- **MUST** alert on high-priority or time-sensitive cases
- **MUST** provide dashboard notifications for system health and performance issues

---

## 13. Testing & Quality Assurance Constraints

### 13.1 Testing Requirements
- **MUST** implement comprehensive unit tests for all services and agents
- **MUST** implement integration tests covering end-to-end workflows
- **MUST** perform load testing before production rollout
- **MUST** test with synthetic and production-like data
- **SHOULD** achieve minimum 80% code coverage for critical paths

### 13.2 Quality Gates
- **MUST** require passing unit tests before merge
- **MUST** require passing integration tests before staging deployment
- **MUST** require manual approval for production deployment
- **MUST** validate agent accuracy against baseline before promoting new models

---

## 14. Rollout & Migration Constraints

### 14.1 Phased Rollout
- **MUST** start with limited dispute types and user groups for pilot
- **MUST** collect feedback and iterate before full rollout
- **MUST** implement gradual rollout to more dispute types and channels
- **MUST** monitor metrics continuously during rollout
- **MUST** have rollback plan for each deployment stage

### 14.2 Success Metrics
- **MUST** track Mean Time to Resolution (MTTR)
- **MUST** track First Contact Resolution (FCR)
- **MUST** track Customer Satisfaction (CSAT/NPS)
- **MUST** track Operational Cost per Case
- **MUST** track AI Accuracy Rate and Confidence Calibration
- **MUST** track Escalation Rate and Override Frequency

---

## 15. Constraint Compliance Matrix

| Category | Critical Constraints | Priority |
|----------|---------------------|----------|
| Platform | Vertex AI Agent Engine, Google ADK, GCP | P0 |
| Security | CMEK encryption, TLS 1.3, RBAC, Audit Logs | P0 |
| Data Flow | DMS → Vector Store, Cases → BigQuery → MLOps | P0 |
| Agent Architecture | Orchestrator → Triage → Analysis, Confidence Thresholds | P0 |
| Integration | API Gateway, Tool Connectors, Circuit Breakers | P0 |
| MLOps | Model Registry, Retraining Triggers, Feedback Loop | P0 |
| UX | 3-min submission, AI recommendations with confidence | P1 |
| Observability | Structured logging, distributed tracing, dashboards | P1 |
| CI/CD | Terraform IaC, automated testing, progressive delivery | P1 |
| Performance | <5s triage, <30s analysis, auto-scaling | P2 |

---

## Document Version
- **Version**: 1.0
- **Last Updated**: 22 November 2025
- **Source**: dispute-resolution-architecture-overview.md
- **Purpose**: Implementation guidance and design compliance validation
