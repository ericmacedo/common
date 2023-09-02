from sqlalchemy.orm import registry

MapperRegistry = registry()
Base = MapperRegistry.generate_base()
