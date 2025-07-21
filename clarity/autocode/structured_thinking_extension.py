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
        """Break down a coding problem into components"""
        
        components = []
        tags = []
        axioms = []
        assumptions = []
        
        # Detect problem type
        if "api" in problem.lower():
            components.append("API integration")
            tags.extend(["api", "integration"])
            axioms.append("APIs should be treated as external dependencies")
            assumptions.append("API will remain stable")
        
        if "database" in problem.lower():
            components.append("Data persistence")
            tags.extend(["database", "persistence"])
            axioms.append("Data consistency is critical")
            assumptions.append("Database schema is optimized")
        
        if "performance" in problem.lower():
            components.append("Performance optimization")
            tags.extend(["performance", "optimization"])
            axioms.append("Premature optimization is root of all evil")
            assumptions.append("Current performance is actually problematic")
        
        if "test" in problem.lower():
            components.append("Testing strategy")
            tags.extend(["testing", "quality"])
            axioms.append("Tests should be readable and maintainable")
        
        # Add context-based components
        if context:
            if context.get("language") == "python":
                components.append("Python-specific implementation")
                tags.append("python")
            
            if context.get("framework"):
                components.append(f"{context['framework']} framework usage")
                tags.append(context["framework"].lower())
        
        # Default components if none detected
        if not components:
            components = ["Implementation logic", "Error handling", "Code structure"]
            tags = ["implementation", "general"]
        
        return {
            "components": components,
            "tags": tags,
            "axioms": axioms,
            "assumptions": assumptions
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