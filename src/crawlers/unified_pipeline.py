"""
Unified news ingestion pipeline.
Orchestrates the complete news processing workflow for multiple sources.
"""
import logging
import time
from typing import List, Type
from sqlalchemy.orm import sessionmaker

from src.database.models_migration import init_db_and_create
from src.crawlers.base_adapter import BaseNewsAdapter
from src.crawlers.processors.nlp_processor import NLPProcessor
from src.crawlers.processors.analysis_processor import AnalysisProcessor
from src.crawlers.adapters.finnhub_adapter import FinnhubAdapter
from src.crawlers.adapters.vnexpress_adapter import VnExpressAdapter


class UnifiedNewsPipeline:
    """Unified pipeline for processing news from multiple sources"""
    
    def __init__(self, database_uri: str = None):
        # Initialize database
        self.engine = init_db_and_create(database_uri)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Initialize processors
        self.nlp_processor = NLPProcessor()
        self.analysis_processor = AnalysisProcessor()
        
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    def process_source(self, adapter: BaseNewsAdapter, **adapter_kwargs) -> tuple:
        """Process news from a single source adapter"""
        session = self.SessionLocal()
        inserted = 0
        skipped = 0
        
        try:
            logging.info(f"Processing news from {adapter.source_name}")
            
            # Fetch news items from source
            news_items = adapter.fetch_news(**adapter_kwargs)
            logging.info(f"Fetched {len(news_items)} items from {adapter.source_name}")
            
            for news_item in news_items:
                try:
                    # Check for duplicates
                    existing = self.analysis_processor.check_duplicate(session, news_item)
                    if existing:
                        logging.info(f"Skipping duplicate: {news_item.headline[:80]}")
                        skipped += 1
                        continue
                    
                    # Summarize content
                    text_to_summarize = news_item.content_raw or news_item.headline
                    content_summary = self.nlp_processor.summarize_text(text_to_summarize)
                    
                    # Store news
                    news = self.analysis_processor.store_news(session, news_item, content_summary)
                    logging.info(f"Stored news id={news.id}: {news_item.headline[:80]}")
                    
                    # Analyze sentiment
                    text_for_sentiment = f"{news_item.headline}. {content_summary}"
                    sentiment_result = self.nlp_processor.analyze_sentiment(text_for_sentiment)
                    
                    # Store analysis
                    analysis = self.analysis_processor.store_analysis(
                        session, news, sentiment_result, content_summary, news_item.metadata
                    )
                    logging.info(f"Stored analysis id={analysis.id} (sentiment={sentiment_result['sentiment']})")
                    
                    inserted += 1
                    time.sleep(0.2)  # Be polite to avoid overwhelming resources
                    
                except Exception as e:
                    logging.error(f"Error processing item from {adapter.source_name}: {e}")
                    session.rollback()
                    continue
            
        finally:
            session.close()
        
        logging.info(f"Completed {adapter.source_name}: Inserted={inserted}, Skipped={skipped}")
        return inserted, skipped
    
    def run_pipeline(self, sources: List[str] = None, **kwargs) -> dict:
        """
        Run the complete pipeline for specified sources
        
        Args:
            sources: List of source names to process. If None, processes all available sources.
            **kwargs: Additional arguments passed to source adapters
        
        Returns:
            Dict with processing statistics for each source
        """
        # Available source adapters
        available_adapters = {
            'finnhub': FinnhubAdapter,
            'vnexpress': VnExpressAdapter
        }
        
        # Determine which sources to process
        if sources is None:
            sources = list(available_adapters.keys())
        
        results = {}
        
        for source_name in sources:
            if source_name not in available_adapters:
                logging.warning(f"Unknown source: {source_name}")
                continue
            
            try:
                # Initialize adapter
                adapter_class = available_adapters[source_name]
                adapter = adapter_class()
                
                # Process source
                inserted, skipped = self.process_source(adapter, **kwargs)
                results[source_name] = {
                    'inserted': inserted,
                    'skipped': skipped,
                    'status': 'success'
                }
                
            except Exception as e:
                logging.error(f"Failed to process {source_name}: {e}")
                results[source_name] = {
                    'inserted': 0,
                    'skipped': 0,
                    'status': 'error',
                    'error': str(e)
                }
        
        # Summary
        total_inserted = sum(r.get('inserted', 0) for r in results.values())
        total_skipped = sum(r.get('skipped', 0) for r in results.values())
        logging.info(f"Pipeline completed. Total: Inserted={total_inserted}, Skipped={total_skipped}")
        
        return results


# Convenience functions
def run_all_sources(**kwargs):
    """Run pipeline for all available sources"""
    pipeline = UnifiedNewsPipeline()
    return pipeline.run_pipeline(**kwargs)


def run_single_source(source_name: str, **kwargs):
    """Run pipeline for a single source"""
    pipeline = UnifiedNewsPipeline()
    return pipeline.run_pipeline(sources=[source_name], **kwargs)


if __name__ == "__main__":
    # Example: run pipeline for all sources
    results = run_all_sources()
    print("Pipeline Results:", results)