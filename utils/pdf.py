from functools import cached_property, reduce
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
from requests import Response
from tika import parser as TikaParser


class PDF:
    def __init__(self, **kwargs):
        """Instanciate a PDF object parsed with Apache's Tika
        Tika Apache -- https://tika.apache.org/
        Tika Python -- https://github.com/chrismattmann/tika-python


        Keyword arguments:
        path (pathlib.Path) -- the path to a valid PDF file
        buffer (requests.Response) -- a buffer object containing a response with the PDF file
        remove_css_selectors (List[str] | str) -- a single or list of CSS selectors to remove from the extracted PDF
        """
        self.__path: Path = kwargs.get("path", None)
        self.__buffer: Response = kwargs.get("buffer", None)

        if self.__path:
            if self.__buffer:
                raise TypeError(
                    "You must provide either 'path' or 'buffer', not both")
            else:
                self.__xml: str = TikaParser.from_file(
                    str(self.__path), xmlContent=True)["content"]
        else:
            if not self.__buffer:
                raise TypeError("You must provide either 'path' or 'buffer'")
            else:
                self.__xml: str = TikaParser.from_buffer(
                    self.__buffer, xmlContent=True)["content"]

        self.__soup: BeautifulSoup = BeautifulSoup(self.__xml, 'html.parser')

        self.__extract: List[str] | str = kwargs.get(
            "remove_css_selectors", "")

        if isinstance(self.__extract, list):
            self.__extract: str = ",".join(self.__extract)
        elif isinstance(self.__extract, str):
            pass
        else:
            raise TypeError("'remove_css_selectors' must be a str or a "
                            "list of str describing a valid CSS selector")
        if self.__extract:
            for tag in self.__soup.select(self.__extract):
                tag.extract()

        self.__remove_repeated_info()

    @cached_property
    def pages(self) -> List[str]:
        """Return a list of string containing the PDF pages"""
        return [
            page.get_text()
            for page in self.__soup.find_all("div", {"class": "page"})]

    @cached_property
    def full_text(self) -> str:
        """Returns the PDF's content as a string"""
        return " ".join(self.pages)

    def __remove_repeated_info(self) -> None:
        """Removes header and footer paragraphs that appears in all pages"""
        pages = self.__soup.find_all("div", {"class": "page"})

        to_remove = reduce(lambda acc, cur: set(acc).intersection(cur), [[
            p.get_text() for p in page.find_all("p") if p.get_text()
        ] for page in pages])

        for paragraph in self.__soup.find_all("p"):
            if paragraph.get_text() in to_remove:
                paragraph.extract()
