## 📁 Purpose
Contains deployment-ready model bundles.
Each bundle is a self-contained inference artifact including:

- Model weights
- Class names
- Preprocessing configuration
- Threshold policy
- Metadata manifest

```text
bundles/
 ├── chestxray_<timestamp>/
 └── latest → symlink to most recent bundle
```

Consumed by 
- Docker inference containers
- FastAPI prediction service