from app import init, predict, health_check

# Initialize model when module is imported
init()

# Export functions for Cerebrium
__all__ = ['init', 'predict', 'health_check'] 