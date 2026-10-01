use std::sync::{Arc, OnceLock, RwLock, Weak};
use crate::kv_router::KvRouter;

static ROUTER: OnceLock<RwLock<Weak<KvRouter>>> = OnceLock::new();

pub fn register(router: &Arc<KvRouter>) -> anyhow::Result<()> {
    if std::env::var("DYN_ALIGNED_ENABLE").as_deref() != Ok("1") {
        return Ok(());
    }
    let mut slot = ROUTER.get_or_init(|| RwLock::new(Weak::new())).write().unwrap();
    if let Some(current) = slot.upgrade() {
        anyhow::ensure!(Arc::ptr_eq(&current, router), "Aligned Frontend requires one model router");
    }
    *slot = Arc::downgrade(router);
    Ok(())
}

pub fn router() -> anyhow::Result<Arc<KvRouter>> {
    ROUTER.get().and_then(|slot| slot.read().unwrap().upgrade())
        .ok_or_else(|| anyhow::anyhow!("Native aligned router is not initialized"))
}
