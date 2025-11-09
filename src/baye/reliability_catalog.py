"""
Reliability Catalog for tools and sources (US-02).

Manages reliability scores r ∈ [0, 1] for different evidence sources,
allowing different tools/sources to have different impact on belief updates.

Features:
- CRUD for reliability profiles
- Source categorization (tool, human, llm, database, etc.)
- Default scores with overrides
- Reliability decay over time (optional)
"""

from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json


# ============================================================================
# Reliability Profile
# ============================================================================

@dataclass
class ReliabilityProfile:
    """
    Reliability profile for a source/tool class.

    Attributes:
        source_name: Identifier (e.g., "web_search", "human:expert")
        source_type: Category (tool, human, llm, database, api)
        reliability: Base reliability score in [0, 1]
        confidence_in_reliability: How confident we are in this score [0, 1]
        description: Human-readable description
        created_at: When profile was created
        updated_at: Last modification time
        metadata: Additional context
    """
    source_name: str
    source_type: str
    reliability: float = 0.5
    confidence_in_reliability: float = 1.0
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate reliability score."""
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError(f"Reliability must be in [0, 1], got {self.reliability}")
        if not 0.0 <= self.confidence_in_reliability <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence_in_reliability}")

    def update_reliability(self, new_reliability: float, reason: str = ""):
        """Update reliability score with audit trail."""
        self.reliability = max(0.0, min(1.0, new_reliability))
        self.updated_at = datetime.now()
        if reason:
            if 'update_history' not in self.metadata:
                self.metadata['update_history'] = []
            self.metadata['update_history'].append({
                'timestamp': datetime.now().isoformat(),
                'old_reliability': self.reliability,
                'new_reliability': new_reliability,
                'reason': reason
            })


# ============================================================================
# Reliability Catalog
# ============================================================================

class ReliabilityCatalog:
    """
    Central catalog of reliability scores for sources/tools.

    Default reliabilities by source type:
    - human:expert: 0.95
    - database: 0.90
    - api:established: 0.85
    - tool:verified: 0.80
    - llm:large: 0.70
    - tool:experimental: 0.50
    - unknown: 0.50
    """

    DEFAULT_RELIABILITIES = {
        "human:expert": 0.95,
        "human:review": 0.90,
        "database:primary": 0.90,
        "database:cache": 0.85,
        "api:established": 0.85,
        "api:third_party": 0.75,
        "tool:verified": 0.80,
        "tool:experimental": 0.50,
        "llm:large": 0.70,
        "llm:small": 0.60,
        "web_search": 0.65,
        "code_execution": 0.85,
        "unknown": 0.50,
    }

    def __init__(self, profiles: Optional[List[ReliabilityProfile]] = None):
        """
        Initialize reliability catalog.

        Args:
            profiles: Optional initial profiles
        """
        self.profiles: Dict[str, ReliabilityProfile] = {}

        # Load provided profiles
        if profiles:
            for profile in profiles:
                self.profiles[profile.source_name] = profile

        # Create default profiles if not provided
        self._ensure_defaults()

    def _ensure_defaults(self):
        """Ensure default profiles exist."""
        for source_name, reliability in self.DEFAULT_RELIABILITIES.items():
            if source_name not in self.profiles:
                source_type = source_name.split(':')[0] if ':' in source_name else source_name
                self.profiles[source_name] = ReliabilityProfile(
                    source_name=source_name,
                    source_type=source_type,
                    reliability=reliability,
                    description=f"Default profile for {source_name}"
                )

    def get_reliability(self, source_name: str) -> float:
        """
        Get reliability score for a source.

        Falls back to source type, then "unknown" if not found.

        Args:
            source_name: Source identifier (e.g., "tool:my_api")

        Returns:
            Reliability score in [0, 1]
        """
        # Exact match
        if source_name in self.profiles:
            return self.profiles[source_name].reliability

        # Try source type (e.g., "tool" from "tool:my_api")
        if ':' in source_name:
            source_type = source_name.split(':')[0]
            if source_type in self.profiles:
                return self.profiles[source_type].reliability

        # Fallback to unknown
        return self.profiles.get("unknown", ReliabilityProfile("unknown", "unknown")).reliability

    def add_profile(self, profile: ReliabilityProfile) -> None:
        """
        Add or update a reliability profile.

        Args:
            profile: Profile to add/update
        """
        self.profiles[profile.source_name] = profile

    def create_profile(self,
                      source_name: str,
                      source_type: str,
                      reliability: float,
                      description: str = "",
                      **metadata) -> ReliabilityProfile:
        """
        Create and add a new reliability profile.

        Args:
            source_name: Source identifier
            source_type: Source category
            reliability: Reliability score
            description: Human-readable description
            **metadata: Additional metadata

        Returns:
            Created profile
        """
        profile = ReliabilityProfile(
            source_name=source_name,
            source_type=source_type,
            reliability=reliability,
            description=description,
            metadata=metadata
        )
        self.add_profile(profile)
        return profile

    def update_reliability(self,
                          source_name: str,
                          new_reliability: float,
                          reason: str = "") -> None:
        """
        Update reliability score for a source.

        Args:
            source_name: Source to update
            new_reliability: New reliability score
            reason: Reason for update (for audit trail)
        """
        if source_name not in self.profiles:
            raise ValueError(f"Source {source_name} not found in catalog")

        self.profiles[source_name].update_reliability(new_reliability, reason)

    def delete_profile(self, source_name: str) -> None:
        """
        Delete a profile.

        Args:
            source_name: Profile to delete
        """
        if source_name in self.profiles:
            del self.profiles[source_name]

    def list_profiles(self, source_type: Optional[str] = None) -> List[ReliabilityProfile]:
        """
        List all profiles, optionally filtered by type.

        Args:
            source_type: Optional filter by source type

        Returns:
            List of profiles
        """
        profiles = list(self.profiles.values())

        if source_type:
            profiles = [p for p in profiles if p.source_type == source_type]

        return sorted(profiles, key=lambda p: p.reliability, reverse=True)

    def get_profile(self, source_name: str) -> Optional[ReliabilityProfile]:
        """
        Get profile by name.

        Args:
            source_name: Source identifier

        Returns:
            Profile if found, None otherwise
        """
        return self.profiles.get(source_name)

    def export_catalog(self) -> List[Dict]:
        """
        Export catalog as list of dicts.

        Returns:
            List of profile dicts
        """
        return [
            {
                'source_name': p.source_name,
                'source_type': p.source_type,
                'reliability': p.reliability,
                'confidence_in_reliability': p.confidence_in_reliability,
                'description': p.description,
                'created_at': p.created_at.isoformat(),
                'updated_at': p.updated_at.isoformat(),
                'metadata': p.metadata
            }
            for p in self.profiles.values()
        ]

    def import_catalog(self, catalog_data: List[Dict]) -> None:
        """
        Import catalog from list of dicts.

        Args:
            catalog_data: List of profile dicts
        """
        for data in catalog_data:
            profile = ReliabilityProfile(
                source_name=data['source_name'],
                source_type=data['source_type'],
                reliability=data['reliability'],
                confidence_in_reliability=data.get('confidence_in_reliability', 1.0),
                description=data.get('description', ''),
                metadata=data.get('metadata', {})
            )
            self.add_profile(profile)

    def save_to_file(self, filepath: str) -> None:
        """
        Save catalog to JSON file.

        Args:
            filepath: Path to save file
        """
        import json
        with open(filepath, 'w') as f:
            json.dump(self.export_catalog(), f, indent=2)

    def load_from_file(self, filepath: str) -> None:
        """
        Load catalog from JSON file.

        Args:
            filepath: Path to load from
        """
        import json
        with open(filepath, 'r') as f:
            catalog_data = json.load(f)
        self.import_catalog(catalog_data)

    def get_statistics(self) -> Dict:
        """
        Get catalog statistics.

        Returns:
            Dict with stats (count by type, avg reliability, etc.)
        """
        profiles = list(self.profiles.values())

        if not profiles:
            return {
                'total_profiles': 0,
                'by_type': {},
                'avg_reliability': 0.0,
                'min_reliability': 0.0,
                'max_reliability': 0.0
            }

        # Count by type
        by_type = {}
        for p in profiles:
            if p.source_type not in by_type:
                by_type[p.source_type] = 0
            by_type[p.source_type] += 1

        # Reliability stats
        reliabilities = [p.reliability for p in profiles]

        return {
            'total_profiles': len(profiles),
            'by_type': by_type,
            'avg_reliability': sum(reliabilities) / len(reliabilities),
            'min_reliability': min(reliabilities),
            'max_reliability': max(reliabilities)
        }


# ============================================================================
# Helper Functions
# ============================================================================

def create_default_catalog() -> ReliabilityCatalog:
    """
    Create a catalog with sensible defaults.

    Returns:
        Initialized ReliabilityCatalog
    """
    return ReliabilityCatalog()


def get_reliability_for_tool(tool_name: str,
                             catalog: Optional[ReliabilityCatalog] = None) -> float:
    """
    Convenience function to get reliability for a tool.

    Args:
        tool_name: Tool identifier
        catalog: Optional catalog (creates default if not provided)

    Returns:
        Reliability score
    """
    if catalog is None:
        catalog = create_default_catalog()

    return catalog.get_reliability(tool_name)
