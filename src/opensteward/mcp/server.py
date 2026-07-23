from mcp.server.fastmcp import FastMCP

from opensteward import __version__
from opensteward.models import ReviewCostFactors,ReviewCostResult
from opensteward.review_cost import calculate_review_cost
from opensteward.settings import get_settings
from opensteward.mcp.policy_capabilities import (
    evaluate_repository_policy,
    repository_policy_resource,
)
from opensteward.mcp.github_capabilities import (
    assess_pull_request,
)
mcp=FastMCP(
    name="OpenSteward",
    stateless_http=True,
    json_response=True
)

@mcp.tool()
def system_status()->dict[str,str]:
    """Return the current OpenSteward system status."""
    settings = get_settings()
    return {
        "name": settings.app_name,
        "version": __version__,
        "environment": settings.environment,
        "stage": "version-1-foundation",
        "mode": "read-only",
    }
    
@mcp.tool()
def estimate_review_cost(
    factors: ReviewCostFactors,
) -> ReviewCostResult:
    """Estimate how much maintainer attention a contribution may require.

    The score measures expected review effort. It does not measure
    contributor skill, trustworthiness, or contribution quality.

    Args:
        factors: Normalized review-cost signals from 0 to 100.

    Returns:
        A weighted score with a readable level and per-factor explanations.
    """

    return calculate_review_cost(factors)
mcp.tool()(evaluate_repository_policy)
mcp.tool()(assess_pull_request)
mcp.resource(
    "steward://repository/policy"
)(repository_policy_resource)

