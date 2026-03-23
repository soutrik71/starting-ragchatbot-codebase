"""
Shared fixtures for the RAG chatbot test suite.

Isolation strategy:
  - VectorStore  → real ChromaDB in pytest's tmp_path (auto-cleaned per test)
  - Anthropic    → MagicMock injected via mocker or passed as mock_anthropic_client
  - SessionManager → in-memory, instantiated fresh per test
"""

import pytest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Sample course text fixtures
# ---------------------------------------------------------------------------

SAMPLE_COURSE_CONTENT = """\
Course Title: Introduction to Python Programming
Course Link: https://www.example.com/courses/python-intro
Course Instructor: Jane Smith

Lesson 0: Getting Started with Python
Lesson Link: https://www.example.com/courses/python-intro/lesson/0
Python is a high-level, interpreted programming language known for its simplicity \
and readability. It was created by Guido van Rossum and first released in 1991. \
Python supports multiple programming paradigms, including procedural, object-oriented, \
and functional programming. It has a comprehensive standard library that includes \
modules for file I/O, system calls, and internet protocols. Python is widely used \
in web development, data science, artificial intelligence, scientific computing, \
and automation scripts.

Lesson 1: Variables and Data Types
Lesson Link: https://www.example.com/courses/python-intro/lesson/1
In Python, variables are dynamically typed, meaning you do not need to declare the \
type before assigning a value. Python supports several built-in data types including \
integers, floats, strings, booleans, lists, tuples, dictionaries, and sets. You can \
check the type of a variable using the type() function. String formatting can be done \
using f-strings, the format() method, or percent-style formatting. Understanding data \
types is fundamental to writing correct and efficient Python programs.

Lesson 2: Control Flow and Loops
Lesson Link: https://www.example.com/courses/python-intro/lesson/2
Control flow in Python uses indentation to define code blocks. The if-elif-else \
statement allows conditional execution. For loops iterate over sequences like lists, \
tuples, or strings. While loops execute as long as a condition is true. Python provides \
break and continue statements to control loop execution. List comprehensions offer a \
concise way to create lists based on existing sequences. The range() function generates \
a sequence of numbers useful in iteration.
"""

SAMPLE_COURSE_CONTENT_2 = """\
Course Title: Advanced Machine Learning
Course Link: https://www.example.com/courses/ml-advanced
Course Instructor: Bob Chen

Lesson 0: Neural Networks Fundamentals
Lesson Link: https://www.example.com/courses/ml-advanced/lesson/0
Neural networks are computational models inspired by biological neural networks. \
They consist of layers of interconnected nodes that process information. The input \
layer receives raw data, hidden layers transform it through learned weights, and the \
output layer produces predictions. Activation functions like ReLU, sigmoid, and tanh \
introduce non-linearity. Backpropagation computes gradients to train the network \
by minimising a loss function over many training examples.

Lesson 1: Deep Learning Architectures
Lesson Link: https://www.example.com/courses/ml-advanced/lesson/1
Deep learning involves neural networks with many hidden layers that learn hierarchical \
data representations. Convolutional Neural Networks specialise in grid-like data such \
as images. Recurrent Neural Networks and LSTMs handle sequential data like text and \
time series. Transformers have revolutionised natural language processing through \
self-attention mechanisms. Transfer learning fine-tunes models pre-trained on large \
datasets for specific downstream tasks.
"""


@pytest.fixture(scope="session")
def sample_course_content():
    return SAMPLE_COURSE_CONTENT


@pytest.fixture(scope="session")
def sample_course_content_2():
    return SAMPLE_COURSE_CONTENT_2


@pytest.fixture
def sample_course_file(tmp_path, sample_course_content):
    """Write sample course text to a temp file and return its path."""
    p = tmp_path / "test_course.txt"
    p.write_text(sample_course_content)
    return str(p)


@pytest.fixture
def sample_course_file_2(tmp_path, sample_course_content_2):
    p = tmp_path / "test_course_2.txt"
    p.write_text(sample_course_content_2)
    return str(p)


@pytest.fixture
def sample_docs_folder(tmp_path, sample_course_content, sample_course_content_2):
    """Temp folder containing two course .txt files."""
    (tmp_path / "course1.txt").write_text(sample_course_content)
    (tmp_path / "course2.txt").write_text(sample_course_content_2)
    return str(tmp_path)


# ---------------------------------------------------------------------------
# VectorStore fixtures (real ChromaDB in tmp_path)
# ---------------------------------------------------------------------------

@pytest.fixture
def vector_store(tmp_path):
    from vector_store import VectorStore
    return VectorStore(
        chroma_path=str(tmp_path / "chroma"),
        embedding_model="all-MiniLM-L6-v2",
        max_results=3,
    )


@pytest.fixture
def populated_vector_store(vector_store, sample_course_file):
    """VectorStore with a single real course already indexed."""
    from document_processor import DocumentProcessor
    processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
    course, chunks = processor.process_course_document(sample_course_file)
    vector_store.add_course_metadata(course)
    vector_store.add_course_content(chunks)
    return vector_store


# ---------------------------------------------------------------------------
# Anthropic mock helpers
# ---------------------------------------------------------------------------

def make_direct_response(text: str):
    """Stub an Anthropic message response that ends without tool use."""
    block = MagicMock()
    block.text = text
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [block]
    return resp


def make_tool_use_response(tool_name: str, tool_input: dict, tool_id: str = "tool_abc123"):
    """Stub an Anthropic response that requests a tool call."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input
    block.id = tool_id
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [block]
    return resp


@pytest.fixture
def mock_anthropic_client():
    """A bare MagicMock suitable for use as an Anthropic client."""
    return MagicMock()
