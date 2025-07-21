"""
Structured thinking extension for AutoCode domain.
Enhances code intelligence with systematic problem-solving approaches.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from loguru import logger

from ..domains.structured_thinking import StructuredThought, ThinkingSession, ThinkingStage
from ..domains.structured_thinking_utils import ThinkingAnalyzer, ThinkingMemoryMapper
from ..domains.persistence import PersistenceDomain

class StructuredThinkingExtension:
    """Adds structured thinking capabilities to AutoCode domain"""
    
    def __init__(self, persistence_domain: PersistenceDomain):
        self.persistence_domain = persistence_domain
        self.thinking_sessions = {}  # In-memory session cache
    
    async def analyze_problem_with_stages(self, problem: str, project_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze a coding problem using structured thinking stages"""
        
        session_id = f"problem_analysis_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        # Stage 1: Problem Definition
        problem_thought = StructuredThought(
            thought_number=1,
            total_expected=5,
            stage=ThinkingStage.PROBLEM_DEFINITION,
            content=f"Problem: {problem}",
            tags=["problem", "analysis", "coding"],
            importance=0.8
        )
        
        # Store problem definition
        await self.persistence_domain.store_structured_thought(problem_thought, session_id)
        
        # Stage 2: Research - Look for similar problems
        research_memories = await self.persistence_domain.retrieve_memories(
            query=problem,
            memory_types=["project_pattern", "session_summary", "solution_synthesis"],
            limit=5,
            min_similarity=0.6
        )
        
        research_content = f"Found {len(research_memories)} similar problems in memory."
        if research_memories:
            research_content += " Key patterns: " + ", ".join([
                mem.get("memory_type", "unknown") for mem in research_memories[:3]
            ])
        
        research_thought = StructuredThought(
            thought_number=2,
            total_expected=5,
            stage=ThinkingStage.RESEARCH,
            content=research_content,
            tags=["research", "patterns", "historical"],
            importance=0.7
        )
        
        await self.persistence_domain.store_structured_thought(research_thought, session_id)
        
        # Stage 3: Analysis - Break down the problem
        analysis_components = self._analyze_problem_components(problem, project_context)
        analysis_thought = StructuredThought(
            thought_number=3,
            total_expected=5,
            stage=ThinkingStage.ANALYSIS,
            content=f"Problem components: {', '.join(analysis_components['components'])}",
            tags=analysis_components['tags'],
            axioms=analysis_components['axioms'],
            assumptions_challenged=analysis_components['assumptions'],
            importance=0.8
        )
        
        await self.persistence_domain.store_structured_thought(analysis_thought, session_id)
        
        # Return structured analysis
        return {
            "session_id": session_id,
            "problem_definition": problem_thought.content,
            "research_findings": research_memories,
            "analysis_components": analysis_components,
            "next_stages": ["synthesis", "conclusion"],
            "thinking_progress": "3/5 stages completed"
        }
    
    def _analyze_problem_components(self, problem: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced problem component analysis with sophisticated pattern recognition.
        
        Similar to memory system's multi-dimensional pattern detection.
        """
        
        components = []
        tags = []
        axioms = []
        assumptions = []
        complexity_indicators = {}
        risk_factors = []
        
        # Multi-dimensional component detection
        component_patterns = self._detect_component_patterns(problem, context)
        
        # Technical architecture components
        architecture_components = self._analyze_architecture_components(problem, context)
        components.extend(architecture_components["components"])
        tags.extend(architecture_components["tags"])
        axioms.extend(architecture_components["axioms"])
        assumptions.extend(architecture_components["assumptions"])
        risk_factors.extend(architecture_components.get("risks", []))
        
        # Business logic components
        business_components = self._analyze_business_components(problem, context)
        components.extend(business_components["components"])
        tags.extend(business_components["tags"])
        axioms.extend(business_components["axioms"])
        assumptions.extend(business_components["assumptions"])
        
        # Quality and process components
        quality_components = self._analyze_quality_components(problem, context)
        components.extend(quality_components["components"])
        tags.extend(quality_components["tags"])
        axioms.extend(quality_components["axioms"])
        assumptions.extend(quality_components["assumptions"])
        
        # Integration and external dependencies
        integration_components = self._analyze_integration_components(problem, context)
        components.extend(integration_components["components"])
        tags.extend(integration_components["tags"])
        axioms.extend(integration_components["axioms"])
        assumptions.extend(integration_components["assumptions"])
        
        # Performance and scalability
        performance_components = self._analyze_performance_components(problem, context)
        components.extend(performance_components["components"])
        tags.extend(performance_components["tags"])
        axioms.extend(performance_components["axioms"])
        assumptions.extend(performance_components["assumptions"])
        
        # Context-aware technology stack analysis
        if context:
            tech_components = self._analyze_technology_stack(problem, context)
            components.extend(tech_components["components"])
            tags.extend(tech_components["tags"])
            axioms.extend(tech_components["axioms"])
            assumptions.extend(tech_components["assumptions"])
        
        # Complexity scoring (like memory system's complexity analysis)
        complexity_indicators = self._calculate_component_complexity(components, problem, context)
        
        # Remove duplicates while preserving order
        components = list(dict.fromkeys(components))
        tags = list(dict.fromkeys(tags))
        axioms = list(dict.fromkeys(axioms))
        assumptions = list(dict.fromkeys(assumptions))
        
        # Default fallback if no components detected
        if not components:
            components = ["Core implementation logic", "Error handling", "Code structure", "Testing strategy"]
            tags = ["implementation", "general", "fallback"]
        
        return {
            "components": components,
            "tags": tags,
            "axioms": axioms,
            "assumptions": assumptions,
            "complexity_indicators": complexity_indicators,
            "risk_factors": risk_factors,
            "component_count": len(components),
            "detection_confidence": min(1.0, len(components) * 0.1 + (len(tags) * 0.05))
        }
    
    def _detect_component_patterns(self, problem: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect high-level component patterns in the problem"""
        patterns = {}
        problem_lower = problem.lower()
        
        # CRUD operations
        crud_indicators = ["create", "read", "update", "delete", "crud", "insert", "select", "modify"]
        crud_score = sum(1 for indicator in crud_indicators if indicator in problem_lower)
        if crud_score > 0:
            patterns["crud_operations"] = {"score": crud_score, "confidence": min(1.0, crud_score * 0.25)}
        
        # Real-time features
        realtime_indicators = ["real-time", "live", "websocket", "streaming", "push", "notification"]
        realtime_score = sum(1 for indicator in realtime_indicators if indicator in problem_lower)
        if realtime_score > 0:
            patterns["realtime_features"] = {"score": realtime_score, "confidence": min(1.0, realtime_score * 0.3)}
        
        # Security concerns
        security_indicators = ["auth", "login", "security", "token", "encrypt", "password", "permission"]
        security_score = sum(1 for indicator in security_indicators if indicator in problem_lower)
        if security_score > 0:
            patterns["security_features"] = {"score": security_score, "confidence": min(1.0, security_score * 0.2)}
        
        return patterns
    
    def _analyze_architecture_components(self, problem: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze architectural components and patterns"""
        components = []
        tags = []
        axioms = []
        assumptions = []
        risks = []
        
        problem_lower = problem.lower()
        
        # Frontend/UI components
        if any(term in problem_lower for term in ["frontend", "ui", "interface", "component", "react", "vue", "angular"]):
            components.append("User interface design and components")
            tags.extend(["frontend", "ui", "components"])
            axioms.append("User experience drives adoption")
            assumptions.append("Users will interact predictably with the interface")
            risks.append("UI complexity can impact performance")
        
        # Backend/API components
        if any(term in problem_lower for term in ["backend", "api", "server", "endpoint", "service"]):
            components.append("Backend API design and implementation")
            tags.extend(["backend", "api", "server"])
            axioms.append("APIs should be versioned and documented")
            assumptions.append("API consumers will handle errors gracefully")
            risks.append("API changes can break client applications")
        
        # Database components
        if any(term in problem_lower for term in ["database", "db", "sql", "nosql", "data", "storage"]):
            components.append("Data modeling and persistence layer")
            tags.extend(["database", "persistence", "data-modeling"])
            axioms.append("Data integrity is non-negotiable")
            assumptions.append("Database performance meets application requirements")
            risks.append("Data migration complexity in production")
        
        # Microservices architecture
        if any(term in problem_lower for term in ["microservice", "distributed", "service", "docker", "kubernetes"]):
            components.append("Microservices architecture and orchestration")
            tags.extend(["microservices", "distributed", "orchestration"])
            axioms.append("Services should be loosely coupled")
            assumptions.append("Network communication is reliable")
            risks.append("Distributed system complexity and latency")
        
        return {
            "components": components,
            "tags": tags,
            "axioms": axioms,
            "assumptions": assumptions,
            "risks": risks
        }
    
    def _analyze_business_components(self, problem: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze business logic and domain-specific components"""
        components = []
        tags = []
        axioms = []
        assumptions = []
        
        problem_lower = problem.lower()
        
        # User management
        if any(term in problem_lower for term in ["user", "account", "profile", "registration", "login"]):
            components.append("User management and authentication")
            tags.extend(["user-management", "authentication"])
            axioms.append("User data privacy is paramount")
            assumptions.append("Users will maintain account security")
        
        # Payment processing
        if any(term in problem_lower for term in ["payment", "billing", "transaction", "money", "purchase"]):
            components.append("Payment processing and financial logic")
            tags.extend(["payments", "financial", "transactions"])
            axioms.append("Financial data requires highest security")
            assumptions.append("Payment providers remain available")
        
        # Content management
        if any(term in problem_lower for term in ["content", "cms", "article", "post", "media"]):
            components.append("Content management and workflow")
            tags.extend(["content", "cms", "workflow"])
            axioms.append("Content versioning prevents data loss")
            assumptions.append("Content editors follow established workflows")
        
        # Notification systems
        if any(term in problem_lower for term in ["notification", "alert", "message", "email", "sms"]):
            components.append("Notification and communication system")
            tags.extend(["notifications", "communication"])
            axioms.append("Critical notifications must be reliable")
            assumptions.append("Users prefer relevant notifications only")
        
        return {
            "components": components,
            "tags": tags,
            "axioms": axioms,
            "assumptions": assumptions
        }
    
    def _analyze_quality_components(self, problem: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze quality assurance and process components"""
        components = []
        tags = []
        axioms = []
        assumptions = []
        
        problem_lower = problem.lower()
        
        # Testing strategy
        if any(term in problem_lower for term in ["test", "testing", "qa", "quality", "unit", "integration"]):
            components.append("Testing strategy and quality assurance")
            tags.extend(["testing", "quality", "qa"])
            axioms.append("Tests should be fast, reliable, and maintainable")
            assumptions.append("Test coverage accurately reflects code quality")
        
        # Monitoring and logging
        if any(term in problem_lower for term in ["monitor", "logging", "metrics", "observability", "debug"]):
            components.append("Monitoring, logging, and observability")
            tags.extend(["monitoring", "logging", "observability"])
            axioms.append("Observability enables rapid problem resolution")
            assumptions.append("Log data will not overwhelm storage")
        
        # Documentation
        if any(term in problem_lower for term in ["document", "docs", "readme", "guide", "manual"]):
            components.append("Documentation and knowledge management")
            tags.extend(["documentation", "knowledge"])
            axioms.append("Code should be self-documenting when possible")
            assumptions.append("Documentation will be maintained current")
        
        # CI/CD pipeline
        if any(term in problem_lower for term in ["deploy", "pipeline", "ci", "cd", "build", "release"]):
            components.append("Continuous integration and deployment")
            tags.extend(["ci-cd", "deployment", "pipeline"])
            axioms.append("Automated deployments reduce human error")
            assumptions.append("Deployment rollback procedures work reliably")
        
        return {
            "components": components,
            "tags": tags,
            "axioms": axioms,
            "assumptions": assumptions
        }
    
    def _analyze_integration_components(self, problem: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze external integrations and dependencies"""
        components = []
        tags = []
        axioms = []
        assumptions = []
        
        problem_lower = problem.lower()
        
        # Third-party APIs
        if any(term in problem_lower for term in ["api", "integration", "external", "third-party", "webhook"]):
            components.append("Third-party API integrations")
            tags.extend(["integration", "external-apis"])
            axioms.append("External APIs should have fallback mechanisms")
            assumptions.append("Third-party services maintain uptime SLAs")
        
        # Authentication providers
        if any(term in problem_lower for term in ["oauth", "sso", "google", "github", "facebook", "auth0"]):
            components.append("External authentication provider integration")
            tags.extend(["oauth", "sso", "auth-providers"])
            axioms.append("Authentication should support multiple providers")
            assumptions.append("OAuth providers maintain security standards")
        
        # Data synchronization
        if any(term in problem_lower for term in ["sync", "synchronize", "import", "export", "migration"]):
            components.append("Data synchronization and migration")
            tags.extend(["data-sync", "migration"])
            axioms.append("Data consistency across systems is critical")
            assumptions.append("Data formats remain stable during migration")
        
        return {
            "components": components,
            "tags": tags,
            "axioms": axioms,
            "assumptions": assumptions
        }
    
    def _analyze_performance_components(self, problem: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance and scalability components"""
        components = []
        tags = []
        axioms = []
        assumptions = []
        
        problem_lower = problem.lower()
        
        # Caching strategy
        if any(term in problem_lower for term in ["cache", "caching", "redis", "memcached", "performance"]):
            components.append("Caching strategy and implementation")
            tags.extend(["caching", "performance"])
            axioms.append("Caching improves performance but adds complexity")
            assumptions.append("Cache invalidation strategies are reliable")
        
        # Database optimization
        if any(term in problem_lower for term in ["optimize", "index", "query", "performance", "slow"]):
            components.append("Database query optimization")
            tags.extend(["database-optimization", "performance"])
            axioms.append("Premature optimization is root of all evil")
            assumptions.append("Performance bottlenecks are accurately identified")
        
        # Load balancing and scaling
        if any(term in problem_lower for term in ["scale", "scaling", "load", "balance", "horizontal"]):
            components.append("Scaling and load distribution")
            tags.extend(["scaling", "load-balancing"])
            axioms.append("Design for scale from the beginning")
            assumptions.append("Traffic patterns are predictable")
        
        return {
            "components": components,
            "tags": tags,
            "axioms": axioms,
            "assumptions": assumptions
        }
    
    def _analyze_technology_stack(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze technology stack specific components"""
        components = []
        tags = []
        axioms = []
        assumptions = []
        
        # Programming language specific components
        language = context.get("language", "").lower()
        if language == "python":
            components.append("Python-specific implementation patterns")
            tags.extend(["python", "language-specific"])
            axioms.append("Python's readability enhances maintainability")
            assumptions.append("Python performance is adequate for use case")
        elif language == "javascript":
            components.append("JavaScript/Node.js implementation")
            tags.extend(["javascript", "nodejs"])
            axioms.append("Asynchronous patterns prevent blocking")
            assumptions.append("npm packages are security-audited")
        
        # Framework specific components
        framework = context.get("framework", "").lower()
        if framework in ["django", "flask", "fastapi"]:
            components.append(f"{framework.title()} framework integration")
            tags.extend([framework, "web-framework"])
            axioms.append("Framework conventions reduce development time")
            assumptions.append("Framework remains actively maintained")
        
        # Cloud platform components
        platform = context.get("platform", "").lower()
        if platform in ["aws", "gcp", "azure"]:
            components.append(f"{platform.upper()} cloud service integration")
            tags.extend([platform, "cloud"])
            axioms.append("Cloud services provide scalability and reliability")
            assumptions.append("Cloud costs remain within budget")
        
        return {
            "components": components,
            "tags": tags,
            "axioms": axioms,
            "assumptions": assumptions
        }
    
    def _calculate_component_complexity(self, components: List[str], problem: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate complexity indicators similar to memory system's analysis"""
        complexity_score = 0.0
        
        # Component count complexity
        component_complexity = min(1.0, len(components) * 0.1)
        complexity_score += component_complexity
        
        # Problem description complexity
        problem_words = len(problem.split())
        description_complexity = min(0.3, problem_words * 0.01)
        complexity_score += description_complexity
        
        # Context complexity
        context_complexity = 0.0
        if context:
            tech_count = len(context.get("technologies", []))
            context_complexity = min(0.2, tech_count * 0.05)
        complexity_score += context_complexity
        
        # Integration complexity
        integration_keywords = ["api", "integration", "external", "third-party"]
        integration_complexity = min(0.2, sum(1 for kw in integration_keywords if kw in problem.lower()) * 0.05)
        complexity_score += integration_complexity
        
        return {
            "total_complexity": min(1.0, complexity_score),
            "component_complexity": component_complexity,
            "description_complexity": description_complexity,
            "context_complexity": context_complexity,
            "integration_complexity": integration_complexity,
            "complexity_level": "high" if complexity_score > 0.7 else "medium" if complexity_score > 0.4 else "low"
        }
    
    async def suggest_next_thinking_stage(self, session_id: str) -> Dict[str, Any]:
        """Suggest the next stage in structured thinking process"""
        
        session = await self.persistence_domain.retrieve_thinking_session(session_id)
        if not session:
            return {"error": "Session not found"}
        
        # Determine current stage
        stages_completed = set(t.stage for t in session.thoughts)
        last_thought = max(session.thoughts, key=lambda t: t.thought_number) if session.thoughts else None
        
        if not last_thought:
            return {"next_stage": "problem_definition", "reason": "No thoughts recorded yet"}
        
        # Stage progression logic
        stage_progression = {
            ThinkingStage.PROBLEM_DEFINITION: ("research", "Gather information about similar problems"),
            ThinkingStage.RESEARCH: ("analysis", "Break down the problem into components"),
            ThinkingStage.ANALYSIS: ("synthesis", "Combine insights to develop solutions"),
            ThinkingStage.SYNTHESIS: ("conclusion", "Make final decisions and create action plan"),
            ThinkingStage.CONCLUSION: ("complete", "Thinking process is complete")
        }
        
        next_stage_info = stage_progression.get(last_thought.stage)
        if next_stage_info:
            next_stage, reason = next_stage_info
            
            # Provide stage-specific guidance
            guidance = self._get_stage_guidance(next_stage, session)
            
            return {
                "next_stage": next_stage,
                "reason": reason,
                "guidance": guidance,
                "current_progress": f"{len(stages_completed)}/5 stages",
                "session_id": session_id
            }
        
        return {"next_stage": "complete", "reason": "All stages completed"}
    
    async def auto_progress_thinking_stage(self, session_id: str, auto_execute: bool = True) -> Dict[str, Any]:
        """
        Automatically progress to next thinking stage with intelligent content generation.
        
        Similar to memory system's auto-retrieval, this automatically generates
        and executes the next stage of structured thinking.
        """
        try:
            session = await self.persistence_domain.retrieve_thinking_session(session_id)
            if not session:
                return {"error": "Session not found"}
            
            # Get next stage suggestion
            next_stage_info = await self.suggest_next_thinking_stage(session_id)
            next_stage = next_stage_info.get("next_stage")
            
            if next_stage == "complete":
                return {"status": "complete", "message": "All thinking stages completed"}
            
            # Auto-generate content for the next stage
            auto_content = await self._auto_generate_stage_content(session, next_stage)
            
            if auto_execute and auto_content:
                # Automatically create and store the next thought
                next_thought_number = len(session.thoughts) + 1
                
                auto_thought = StructuredThought(
                    thought_number=next_thought_number,
                    total_expected=5,
                    stage=ThinkingStage(next_stage),
                    content=auto_content["content"],
                    tags=auto_content.get("tags", []),
                    axioms=auto_content.get("axioms", []),
                    assumptions_challenged=auto_content.get("assumptions", []),
                    importance=auto_content.get("importance", 0.7),
                    metadata={"auto_generated": True, "generation_confidence": auto_content.get("confidence", 0.7)}
                )
                
                # Store the auto-generated thought
                await self.persistence_domain.store_structured_thought(auto_thought, session_id)
                
                return {
                    "status": "auto_progressed",
                    "stage": next_stage,
                    "thought_number": next_thought_number,
                    "content": auto_content["content"],
                    "confidence": auto_content.get("confidence", 0.7),
                    "auto_generated": True,
                    "next_stage_available": next_stage != "conclusion"
                }
            else:
                return {
                    "status": "suggestion_ready",
                    "stage": next_stage,
                    "suggested_content": auto_content,
                    "guidance": next_stage_info.get("guidance", {})
                }
                
        except Exception as e:
            logger.error(f"Error in auto-progression: {e}")
            return {"error": f"Auto-progression failed: {e}"}
    
    async def _auto_generate_stage_content(self, session: ThinkingSession, stage: str) -> Dict[str, Any]:
        """
        Auto-generate intelligent content for the next thinking stage.
        
        Uses memory system patterns and project context for intelligent generation.
        """
        stage_generators = {
            "research": self._generate_research_content,
            "analysis": self._generate_analysis_content,
            "synthesis": self._generate_synthesis_content,
            "conclusion": self._generate_conclusion_content
        }
        
        generator = stage_generators.get(stage)
        if generator:
            return await generator(session)
        
        return {"content": f"Continue with {stage} stage", "confidence": 0.5}
    
    async def _generate_research_content(self, session: ThinkingSession) -> Dict[str, Any]:
        """Generate research stage content using memory system"""
        problem_thoughts = [t for t in session.thoughts if t.stage == ThinkingStage.PROBLEM_DEFINITION]
        if not problem_thoughts:
            return {"content": "Define the problem first", "confidence": 0.3}
        
        problem_content = problem_thoughts[0].content
        
        # Use memory system to find similar problems
        similar_memories = await self.persistence_domain.retrieve_memories(
            query=problem_content,
            memory_types=["project_pattern", "solution_synthesis", "session_summary"],
            limit=5,
            min_similarity=0.6
        )
        
        if similar_memories:
            research_insights = []
            for memory in similar_memories:
                insight = f"Found similar case: {memory.get('content', '')[:100]}..."
                research_insights.append(insight)
            
            content = f"Research findings from memory system:\n" + "\n".join(research_insights[:3])
            confidence = min(0.9, 0.6 + (len(similar_memories) * 0.05))
        else:
            content = f"Research needed for: {problem_content[:100]}... No similar patterns found in memory."
            confidence = 0.6
        
        return {
            "content": content,
            "tags": ["research", "memory-driven", "pattern-matching"],
            "confidence": confidence,
            "importance": 0.8
        }
    
    async def _generate_analysis_content(self, session: ThinkingSession) -> Dict[str, Any]:
        """Generate analysis stage content based on problem and research"""
        research_thoughts = [t for t in session.thoughts if t.stage == ThinkingStage.RESEARCH]
        problem_thoughts = [t for t in session.thoughts if t.stage == ThinkingStage.PROBLEM_DEFINITION]
        
        if not research_thoughts:
            return {"content": "Complete research stage first", "confidence": 0.3}
        
        # Extract components from problem and research
        all_tags = set()
        for thought in session.thoughts:
            all_tags.update(thought.tags)
        
        # Enhanced component analysis using tags and content
        components = []
        assumptions = []
        
        if "api" in str(all_tags).lower() or "api" in str([t.content for t in session.thoughts]).lower():
            components.append("API design and integration")
            assumptions.append("API endpoints will remain stable")
        
        if "database" in str(all_tags).lower() or "database" in str([t.content for t in session.thoughts]).lower():
            components.append("Database schema and queries")
            assumptions.append("Database performance is adequate")
        
        if not components:
            components = ["Core implementation logic", "Error handling", "Testing strategy"]
        
        content = f"Analysis reveals key components: {', '.join(components)}"
        
        return {
            "content": content,
            "tags": ["analysis", "component-breakdown"] + list(all_tags)[:3],
            "assumptions": assumptions,
            "confidence": 0.8,
            "importance": 0.8
        }
    
    async def _generate_synthesis_content(self, session: ThinkingSession) -> Dict[str, Any]:
        """Generate synthesis stage content combining insights"""
        analysis_thoughts = [t for t in session.thoughts if t.stage == ThinkingStage.ANALYSIS]
        
        if not analysis_thoughts:
            return {"content": "Complete analysis stage first", "confidence": 0.3}
        
        # Synthesize from all previous stages
        key_insights = []
        for thought in session.thoughts:
            if len(thought.content) > 50:  # Focus on substantial thoughts
                key_insights.append(thought.content[:80] + "...")
        
        content = f"Synthesis of key insights:\n" + "\n".join(f"- {insight}" for insight in key_insights[:3])
        content += "\n\nRecommended approach: Implement in phases with testing at each stage."
        
        return {
            "content": content,
            "tags": ["synthesis", "solution-approach", "phased-implementation"],
            "axioms": ["Iterative development reduces risk"],
            "confidence": 0.8,
            "importance": 0.9
        }
    
    async def _generate_conclusion_content(self, session: ThinkingSession) -> Dict[str, Any]:
        """Generate conclusion stage with action plan"""
        synthesis_thoughts = [t for t in session.thoughts if t.stage == ThinkingStage.SYNTHESIS]
        
        if not synthesis_thoughts:
            return {"content": "Complete synthesis stage first", "confidence": 0.3}
        
        # Generate concrete action plan
        content = f"""Final decision and action plan:
1. Implement core functionality first
2. Add advanced features incrementally  
3. Test thoroughly at each phase
4. Monitor performance and optimize

Next immediate steps:
- Set up development environment
- Create project structure
- Implement MVP features
- Establish testing framework"""
        
        return {
            "content": content,
            "tags": ["conclusion", "action-plan", "next-steps"],
            "axioms": ["Start simple, iterate quickly"],
            "confidence": 0.9,
            "importance": 1.0
        }
    
    def _get_stage_guidance(self, stage: str, session: ThinkingSession) -> Dict[str, Any]:
        """Provide specific guidance for each thinking stage"""
        
        guidance_map = {
            "research": {
                "focus": "Look for similar problems, existing solutions, and relevant patterns",
                "questions": [
                    "What similar problems have been solved before?",
                    "What patterns or frameworks apply here?",
                    "What constraints or requirements must be considered?"
                ],
                "memory_queries": [
                    f"similar problem {session.title}",
                    "best practices implementation",
                    "common patterns solution"
                ]
            },
            "analysis": {
                "focus": "Break down the problem into manageable components",
                "questions": [
                    "What are the core components of this problem?",
                    "What assumptions am I making?", 
                    "What are the potential risks or challenges?"
                ],
                "considerations": [
                    "Technical complexity",
                    "Resource requirements",
                    "Time constraints"
                ]
            },
            "synthesis": {
                "focus": "Combine research and analysis to develop solution approaches",
                "questions": [
                    "What solution approach best fits the constraints?",
                    "How can different insights be combined?",
                    "What trade-offs need to be made?"
                ],
                "output_format": "Prioritized list of solution approaches"
            },
            "conclusion": {
                "focus": "Make final decisions and create actionable plans",
                "questions": [
                    "Which approach will be implemented?",
                    "What are the next concrete steps?",
                    "How will success be measured?"
                ],
                "deliverables": [
                    "Implementation plan",
                    "Success criteria",
                    "Risk mitigation strategies"
                ]
            }
        }
        
        return guidance_map.get(stage, {"focus": "Continue structured thinking process"})
    
    async def track_assumption_evolution(self, session_id: str) -> Dict[str, Any]:
        """Track how assumptions change throughout thinking process"""
        
        session = await self.persistence_domain.retrieve_thinking_session(session_id)
        if not session:
            return {"error": "Session not found"}
        
        assumption_evolution = []
        all_assumptions = set()
        
        for thought in sorted(session.thoughts, key=lambda t: t.thought_number):
            # Track new assumptions introduced
            new_assumptions = set(thought.assumptions_challenged) - all_assumptions
            if new_assumptions:
                assumption_evolution.append({
                    "thought_number": thought.thought_number,
                    "stage": thought.stage.value,
                    "action": "challenged",
                    "assumptions": list(new_assumptions),
                    "context": thought.content[:100] + "..." if len(thought.content) > 100 else thought.content
                })
            
            all_assumptions.update(thought.assumptions_challenged)
        
        # Analyze assumption patterns
        assumption_frequency = {}
        for thought in session.thoughts:
            for assumption in thought.assumptions_challenged:
                assumption_frequency[assumption] = assumption_frequency.get(assumption, 0) + 1
        
        return {
            "evolution_timeline": assumption_evolution,
            "total_assumptions_challenged": len(all_assumptions),
            "most_challenged_assumptions": sorted(
                assumption_frequency.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5],
            "assumptions_by_stage": self._group_assumptions_by_stage(session.thoughts)
        }
    
    def _group_assumptions_by_stage(self, thoughts: List[StructuredThought]) -> Dict[str, List[str]]:
        """Group assumptions by thinking stage"""
        
        assumptions_by_stage = {}
        for thought in thoughts:
            stage_key = thought.stage.value
            if stage_key not in assumptions_by_stage:
                assumptions_by_stage[stage_key] = []
            assumptions_by_stage[stage_key].extend(thought.assumptions_challenged)
        
        # Remove duplicates within each stage
        for stage in assumptions_by_stage:
            assumptions_by_stage[stage] = list(set(assumptions_by_stage[stage]))
        
        return assumptions_by_stage
    
    async def generate_coding_action_plan(self, session_id: str) -> Dict[str, Any]:
        """Generate concrete coding action plan from thinking session"""
        
        session = await self.persistence_domain.retrieve_thinking_session(session_id)
        if not session:
            return {"error": "Session not found"}
        
        # Extract actionable items from different stages
        problem_definition = self._extract_problem_definition(session)
        research_insights = self._extract_research_insights(session)
        analysis_components = self._extract_analysis_components(session)
        synthesis_solutions = self._extract_synthesis_solutions(session)
        conclusion_decisions = self._extract_conclusion_decisions(session)
        
        # Generate action plan
        action_plan = {
            "session_id": session_id,
            "problem_statement": problem_definition,
            "research_foundation": research_insights,
            "implementation_components": analysis_components,
            "solution_approach": synthesis_solutions,
            "action_items": conclusion_decisions,
            "next_steps": self._generate_next_steps(session),
            "success_criteria": self._define_success_criteria(session),
            "risk_mitigation": self._identify_risks_and_mitigation(session)
        }
        
        # Store action plan as memory
        plan_memory_id = await self.persistence_domain.store_memory(
            memory_type="solution_synthesis",
            content=f"Action plan for {session.title}: {len(action_plan['action_items'])} action items identified",
            importance=0.9,
            metadata={
                "session_id": session_id,
                "structured_thinking_output": True,
                "plan_type": "coding_implementation",
                "total_action_items": len(action_plan.get("action_items", []))
            }
        )
        
        action_plan["plan_memory_id"] = plan_memory_id
        return action_plan
    
    def _extract_problem_definition(self, session: ThinkingSession) -> str:
        """Extract clear problem definition from session"""
        problem_thoughts = [t for t in session.thoughts if t.stage == ThinkingStage.PROBLEM_DEFINITION]
        if problem_thoughts:
            return problem_thoughts[0].content
        return "Problem definition not clearly established"
    
    def _extract_research_insights(self, session: ThinkingSession) -> List[str]:
        """Extract key research insights"""
        research_thoughts = [t for t in session.thoughts if t.stage == ThinkingStage.RESEARCH]
        insights = []
        
        for thought in research_thoughts:
            # Extract key insights (simplified implementation)
            if "pattern" in thought.content.lower():
                insights.append("Relevant patterns identified")
            if "similar" in thought.content.lower():
                insights.append("Similar problems found in memory")
            if "framework" in thought.content.lower():
                insights.append("Applicable frameworks identified")
        
        return insights if insights else ["Research stage incomplete"]
    
    def _extract_analysis_components(self, session: ThinkingSession) -> List[str]:
        """Extract implementation components from analysis"""
        analysis_thoughts = [t for t in session.thoughts if t.stage == ThinkingStage.ANALYSIS]
        components = []
        
        for thought in analysis_thoughts:
            # Extract components from tags and content
            components.extend(thought.tags)
            
            # Look for component indicators in content
            if "component" in thought.content.lower():
                # Simple extraction - in real implementation, could use NLP
                words = thought.content.split()
                for i, word in enumerate(words):
                    if word.lower() == "component" and i > 0:
                        components.append(words[i-1])
        
        return list(set(components)) if components else ["Implementation components need analysis"]
    
    def _extract_synthesis_solutions(self, session: ThinkingSession) -> List[str]:
        """Extract solution approaches from synthesis stage"""
        synthesis_thoughts = [t for t in session.thoughts if t.stage == ThinkingStage.SYNTHESIS]
        solutions = []
        
        for thought in synthesis_thoughts:
            # Extract solution approaches
            solutions.append(thought.content)
        
        return solutions if solutions else ["Solution synthesis pending"]
    
    def _extract_conclusion_decisions(self, session: ThinkingSession) -> List[str]:
        """Extract concrete action items from conclusions"""
        conclusion_thoughts = [t for t in session.thoughts if t.stage == ThinkingStage.CONCLUSION]
        decisions = []
        
        for thought in conclusion_thoughts:
            # Extract decisions and action items
            decisions.append(thought.content)
        
        return decisions if decisions else ["Final decisions pending"]
    
    def _generate_next_steps(self, session: ThinkingSession) -> List[str]:
        """Generate concrete next steps"""
        return [
            "Review and validate the action plan",
            "Set up development environment if needed",
            "Begin implementation of core components",
            "Implement testing strategy",
            "Monitor progress and adjust as needed"
        ]
    
    def _define_success_criteria(self, session: ThinkingSession) -> List[str]:
        """Define success criteria based on thinking process"""
        return [
            "Problem requirements are met",
            "Implementation follows identified patterns", 
            "Code quality standards are maintained",
            "Solution is properly tested"
        ]
    
    def _identify_risks_and_mitigation(self, session: ThinkingSession) -> Dict[str, str]:
        """Identify risks and mitigation strategies"""
        risks = {}
        
        # Analyze assumptions for potential risks
        for thought in session.thoughts:
            for assumption in thought.assumptions_challenged:
                risk_key = f"Assumption: {assumption}"
                risks[risk_key] = "Validate assumption before proceeding"
        
        # Add general coding risks
        risks["Technical complexity"] = "Break down into smaller, manageable tasks"
        risks["Time constraints"] = "Prioritize core functionality first"
        risks["Integration issues"] = "Test integrations early and frequently"
        
        return risks
    
    async def suggest_proactive_thinking(self, context: Dict[str, Any], limit: int = 3) -> Dict[str, Any]:
        """
        Proactively suggest structured thinking opportunities based on context.
        
        Similar to memory system's suggest_memory_queries, this analyzes context
        and suggests when structured thinking would be beneficial.
        """
        try:
            suggestions = []
            reasoning = []
            
            # Analyze current context for thinking opportunities
            current_task = context.get("current_task", "")
            project_context = context.get("project_context", {})
            recent_activity = context.get("recent_activity", [])
            files_accessed = context.get("files_accessed", [])
            commands_executed = context.get("commands_executed", [])
            
            # Suggestion 1: Task complexity analysis
            if current_task:
                complexity_score = self._analyze_task_complexity(current_task, context)
                if complexity_score >= 0.7:
                    suggestions.append({
                        "type": "complex_task_analysis",
                        "priority": "high",
                        "suggestion": f"Break down complex task: '{current_task[:50]}...'",
                        "thinking_stages": ["problem_definition", "analysis", "synthesis"],
                        "confidence": complexity_score,
                        "estimated_time": "15-20 minutes",
                        "benefits": ["Better problem understanding", "Systematic approach", "Reduced implementation risks"]
                    })
                    reasoning.append(f"Task complexity score {complexity_score:.2f} indicates structured thinking would be beneficial")
            
            # Suggestion 2: Multi-file project analysis
            if len(files_accessed) >= 3:
                file_patterns = self._analyze_file_patterns(files_accessed)
                if file_patterns["complexity"] > 0.6:
                    suggestions.append({
                        "type": "architecture_analysis",
                        "priority": "medium",
                        "suggestion": f"Analyze architecture across {len(files_accessed)} files",
                        "thinking_stages": ["research", "analysis", "synthesis"],
                        "confidence": file_patterns["complexity"],
                        "estimated_time": "10-15 minutes",
                        "benefits": ["System understanding", "Pattern recognition", "Refactoring opportunities"]
                    })
                    reasoning.append(f"Multi-file access pattern suggests architectural thinking needed")
            
            # Suggestion 3: Error pattern analysis
            error_commands = [cmd for cmd in commands_executed if any(err in cmd.lower() for err in ["error", "failed", "exception"])]
            if len(error_commands) >= 2:
                suggestions.append({
                    "type": "debugging_strategy",
                    "priority": "high",
                    "suggestion": "Systematic debugging approach for recurring errors",
                    "thinking_stages": ["problem_definition", "research", "analysis"],
                    "confidence": 0.8,
                    "estimated_time": "10-12 minutes", 
                    "benefits": ["Root cause identification", "Prevention strategies", "Learning from errors"]
                })
                reasoning.append(f"Multiple error patterns detected - systematic debugging recommended")
            
            # Suggestion 4: Learning opportunity detection
            new_technologies = self._detect_new_technologies(project_context, recent_activity)
            if new_technologies:
                suggestions.append({
                    "type": "learning_consolidation",
                    "priority": "medium",
                    "suggestion": f"Consolidate learning about {', '.join(new_technologies[:2])}",
                    "thinking_stages": ["research", "analysis", "synthesis"],
                    "confidence": 0.7,
                    "estimated_time": "8-12 minutes",
                    "benefits": ["Knowledge consolidation", "Pattern recognition", "Future reference"]
                })
                reasoning.append(f"New technologies detected: {', '.join(new_technologies)} - learning consolidation recommended")
            
            # Suggestion 5: Decision point analysis
            decision_indicators = self._detect_decision_points(current_task, recent_activity)
            if decision_indicators["count"] > 0:
                suggestions.append({
                    "type": "decision_analysis",
                    "priority": "high" if decision_indicators["complexity"] > 0.8 else "medium",
                    "suggestion": f"Analyze {decision_indicators['count']} pending decisions systematically",
                    "thinking_stages": ["analysis", "synthesis", "conclusion"],
                    "confidence": decision_indicators["complexity"],
                    "estimated_time": "12-18 minutes",
                    "benefits": ["Clear decision criteria", "Risk assessment", "Confident choices"]
                })
                reasoning.append(f"Decision points detected - structured analysis recommended")
            
            # Sort suggestions by priority and confidence
            priority_order = {"high": 3, "medium": 2, "low": 1}
            suggestions.sort(key=lambda x: (priority_order.get(x["priority"], 0), x["confidence"]), reverse=True)
            
            return {
                "suggestions": suggestions[:limit],
                "total_found": len(suggestions),
                "context_analysis": {
                    "task_complexity": self._analyze_task_complexity(current_task, context) if current_task else 0,
                    "file_complexity": len(files_accessed),
                    "command_patterns": len(commands_executed),
                    "decision_points": decision_indicators.get("count", 0) if 'decision_indicators' in locals() else 0
                },
                "reasoning": reasoning[:3],
                "proactive_thinking_enabled": True
            }
            
        except Exception as e:
            logger.error(f"Error generating proactive thinking suggestions: {e}")
            return {"error": f"Suggestion generation failed: {e}", "suggestions": []}
    
    def _analyze_task_complexity(self, task: str, context: Dict[str, Any]) -> float:
        """Analyze task complexity for thinking suggestion scoring"""
        if not task:
            return 0.0
            
        complexity_score = 0.0
        
        # Length and detail indicators
        if len(task.split()) > 10:
            complexity_score += 0.2
        if len(task.split()) > 20:
            complexity_score += 0.1
            
        # Technical complexity keywords
        complex_keywords = ["implement", "design", "architect", "optimize", "integrate", "migrate", "refactor"]
        for keyword in complex_keywords:
            if keyword in task.lower():
                complexity_score += 0.1
                
        # Multiple component indicators
        multi_indicators = ["and", "also", "including", "with", "plus", "furthermore"]
        for indicator in multi_indicators:
            if indicator in task.lower():
                complexity_score += 0.05
                
        # Project context complexity
        project_context = context.get("project_context", {})
        if len(project_context.get("technologies", [])) > 2:
            complexity_score += 0.1
        if project_context.get("scale") == "enterprise":
            complexity_score += 0.1
            
        return min(complexity_score, 1.0)
    
    def _analyze_file_patterns(self, files_accessed: List[str]) -> Dict[str, Any]:
        """Analyze patterns in accessed files"""
        if not files_accessed:
            return {"complexity": 0.0, "patterns": []}
            
        file_types = set()
        directories = set()
        
        for file_path in files_accessed:
            if '.' in file_path:
                file_types.add(file_path.split('.')[-1])
            if '/' in file_path:
                directories.add('/'.join(file_path.split('/')[:-1]))
                
        complexity = min(1.0, (len(file_types) * 0.1) + (len(directories) * 0.05))
        
        return {
            "complexity": complexity,
            "file_types": list(file_types),
            "directories": list(directories),
            "patterns": ["multi_language" if len(file_types) > 2 else "focused"]
        }
    
    def _detect_new_technologies(self, project_context: Dict[str, Any], recent_activity: List[str]) -> List[str]:
        """Detect new technologies or frameworks being learned"""
        technologies = []
        
        # Check project context
        if project_context:
            frameworks = project_context.get("frameworks", [])
            languages = project_context.get("languages", [])
            technologies.extend(frameworks + languages)
            
        # Check recent activity for technology mentions
        tech_keywords = ["react", "vue", "angular", "django", "flask", "fastapi", "docker", "kubernetes", "aws", "gcp"]
        for activity in recent_activity:
            for tech in tech_keywords:
                if tech in activity.lower() and tech not in technologies:
                    technologies.append(tech)
                    
        return list(set(technologies))[:5]  # Return unique, limited list
    
    def _detect_decision_points(self, task: str, recent_activity: List[str]) -> Dict[str, Any]:
        """Detect pending decisions that need analysis"""
        decision_indicators = ["should I", "which", "how to", "best way", "choose", "decide", "option"]
        
        decision_count = 0
        complexity_factors = 0
        
        all_text = (task or "") + " " + " ".join(recent_activity)
        
        for indicator in decision_indicators:
            if indicator in all_text.lower():
                decision_count += 1
                
        # Complexity factors
        if "vs" in all_text or "versus" in all_text:
            complexity_factors += 0.2
        if "pros and cons" in all_text.lower():
            complexity_factors += 0.3
        if "trade-off" in all_text.lower():
            complexity_factors += 0.2
            
        return {
            "count": decision_count,
            "complexity": min(1.0, complexity_factors + (decision_count * 0.1))
        }
    
    async def auto_trigger_thinking_from_context(self, context: Dict[str, Any], threshold: float = 0.8) -> Dict[str, Any]:
        """
        Automatically trigger structured thinking based on context analysis.
        
        Similar to memory system's automatic memory consultation.
        """
        try:
            # Get proactive suggestions
            suggestions = await self.suggest_proactive_thinking(context, limit=5)
            
            if not suggestions.get("suggestions"):
                return {"status": "no_triggers", "message": "No structured thinking opportunities detected"}
            
            # Find highest priority, high-confidence suggestions
            high_confidence_suggestions = [
                s for s in suggestions["suggestions"] 
                if s["confidence"] >= threshold and s["priority"] == "high"
            ]
            
            if not high_confidence_suggestions:
                return {
                    "status": "below_threshold",
                    "suggestions": suggestions["suggestions"][:2],
                    "message": f"Suggestions available but below confidence threshold {threshold}"
                }
            
            # Auto-trigger the highest confidence suggestion
            best_suggestion = high_confidence_suggestions[0]
            
            # Create initial structured thinking session
            session_title = f"Auto-analysis: {best_suggestion['suggestion']}"
            problem_content = f"Automatically detected need for {best_suggestion['type']}: {best_suggestion['suggestion']}"
            
            analysis_result = await self.analyze_problem_with_stages(
                problem=problem_content,
                project_context=context.get("project_context", {})
            )
            
            if analysis_result and not analysis_result.get("error"):
                return {
                    "status": "auto_triggered",
                    "session_id": analysis_result["session_id"],
                    "suggestion_type": best_suggestion["type"],
                    "confidence": best_suggestion["confidence"],
                    "estimated_time": best_suggestion["estimated_time"],
                    "benefits": best_suggestion["benefits"],
                    "auto_generated": True
                }
            else:
                return {"status": "trigger_failed", "error": analysis_result.get("error", "Unknown error")}
                
        except Exception as e:
            logger.error(f"Error in auto-trigger from context: {e}")
            return {"error": f"Auto-trigger failed: {e}"}