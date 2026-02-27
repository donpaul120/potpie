import asyncio
from celery import Task
from contextlib import asynccontextmanager
from app.core.database import SessionLocal
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseTask(Task):
    _db = None

    @property
    def db(self):
        if self._db is None:
            self._db = SessionLocal()
        return self._db

    @asynccontextmanager
    async def async_db(self):
        """
        Provides an async session with a fresh connection for Celery tasks.

        This creates a non-pooled connection to avoid asyncpg Future binding issues
        when tasks share the same event loop but have different coroutine contexts.

        Usage:
            async with self.async_db() as session:
                result = await session.execute(query)
                await session.commit()
        """
        from app.core.database import create_celery_async_session

        try:
            task_id = self.request.id if self.request else "test"
        except (AttributeError, TypeError):
            task_id = "test"

        logger.debug("Creating fresh async DB connection", task_id=task_id)
        async_session, engine = create_celery_async_session()

        try:
            yield async_session
            logger.debug(
                "Async DB session operation completed successfully", task_id=task_id
            )
        except Exception:
            logger.exception("Error during async DB operation", task_id=task_id)
            raise
        finally:
            try:
                await async_session.close()
                if engine is not None:
                    await engine.dispose()
                logger.debug(
                    "Async DB connection closed and engine disposed", task_id=task_id
                )
            except Exception:
                logger.exception("Error during connection cleanup", task_id=task_id)

    def run_async(self, coro):
        """
        Run the given coroutine in a fresh event loop.

        Using asyncio.run() (fresh loop per task) rather than a long-lived
        reused loop ensures proper asyncio context isolation. Each task gets
        its own Context, so OpenTelemetry span tokens created via
        start_as_current_span() are always detached in the same Context they
        were attached in — preventing the 'Token was created in a different
        Context' ValueError that occurs when pydantic-ai sub-tasks inherit
        a shared accumulated context from a long-lived loop.
        """
        return asyncio.run(coro)

    def on_success(self, retval, task_id, args, kwargs):
        try:
            status = "cancelled" if retval is False else "completed successfully"
            logger.info("Task completed", task_id=task_id, status=status)
        finally:
            if self._db:
                self._db.close()  # Returns to pool
                self._db = None

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called on task failure."""
        # exc is already an exception object from on_failure
        logger.error(
            "Task failed",
            task_id=task_id,
            error=str(exc),
            exc_info=einfo.exc_info if einfo else None,
        )
        if self._db:
            self._db.close()
            self._db = None

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called on task retry."""
        logger.warning("Task retrying", task_id=task_id, error=str(exc))
