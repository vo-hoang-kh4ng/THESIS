# youtube_search_tool.py
from crewai_tools import YoutubeVideoSearchTool

class MyYoutubeSearchTool(YoutubeVideoSearchTool):
    def run(self, **kwargs):
        """
        Custom implementation of the run method for YoutubeVideoSearchTool.
        Ensures that the search_query is a string, extracting the 'description' field if it's a dictionary.

        Args:
            **kwargs: Arguments passed to the tool, including 'search_query' and optionally 'youtube_video_url'.

        Returns:
            The result of the parent class's run method (typically a string with search results or video content).
        """
        if 'search_query' in kwargs and isinstance(kwargs['search_query'], dict):
            kwargs['search_query'] = kwargs['search_query'].get('description', str(kwargs['search_query']))
        return super().run(**kwargs)