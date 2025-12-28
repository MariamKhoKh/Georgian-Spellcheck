import re
import json
import bz2
import xml.etree.ElementTree as ET
from collections import Counter
from typing import List, Set, Dict
from pathlib import Path
import requests
from tqdm import tqdm

GEORGIAN_PATTERN = re.compile(r'[ა-ჰ]+')
VALID_WORD_PATTERN = re.compile(r'^[ა-ჰ]+$')


class GeorgianWikipediaProcessor:
    """
    downloads and processes georgian wikipedia dump to extract vocabulary.
    """
    
    def __init__(self, output_dir: str = 'data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.words = set()
        self.char_freq = Counter()
        
        self.dump_url = "https://dumps.wikimedia.org/kawiki/latest/kawiki-latest-pages-articles.xml.bz2"
        self.dump_file = self.output_dir / "kawiki-latest-pages-articles.xml.bz2"
        
    def download_wikipedia_dump(self) -> bool:
        """
        download georgian wikipedia dump (compressed).
        """
        if self.dump_file.exists():
            print(f"Wikipedia dump already exists at {self.dump_file}")
            return True
        
        print(f"Downloading Georgian wikipedia dump...")
        print(f"URL: {self.dump_url}")
        
        try:
            response = requests.get(self.dump_url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(self.dump_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"Download complete: {self.dump_file}")
            return True
            
        except Exception as e:
            print(f"Download failed: {e}")
            print("\nAlternative: Download manually from:")
            print("https://dumps.wikimedia.org/kawiki/latest/")
            print(f"Save as: {self.dump_file}")
            return False
    
    def extract_text_from_dump(self, max_articles: int = None) -> List[str]:
        """
        extract text from wikipedia xml dump.
        returns list of article texts.
        """
        print("extracting text from wikipedia dump...")
        texts = []
        article_count = 0
        
        try:
            # open compressed file
            with bz2.open(self.dump_file, 'rt', encoding='utf-8') as f:
                
                # parse xml iteratively to save memory
                context = ET.iterparse(f, events=('start', 'end'))
                context = iter(context)
                
                in_text = False
                current_text = []
                
                for event, elem in context:
                    tag = elem.tag.split('}')[-1]  # remove namespace
                    
                    if event == 'start' and tag == 'text':
                        in_text = True
                        current_text = []
                    
                    elif event == 'end' and tag == 'text':
                        in_text = False
                        if elem.text:
                            # clean wikipedia markup
                            text = self.clean_wikipedia_text(elem.text)
                            if text:
                                texts.append(text)
                                article_count += 1
                                
                                if article_count % 1000 == 0:
                                    print(f"  Processed {article_count} articles, "
                                          f"found {len(self.words)} unique words...")
                                
                                if max_articles and article_count >= max_articles:
                                    break
                        
                        # clear element to save memory
                        elem.clear()
                
                print(f"Extracted text from {article_count} articles")
                return texts
                
        except Exception as e:
            print(f"Error parsing dump: {e}")
            return []
    
    def clean_wikipedia_text(self, text: str) -> str:
        """
        remove wikipedia markup and extract clean text.
        """
        # remove templates {{...}}
        text = re.sub(r'\{\{[^}]+\}\}', '', text)
        # remove links [[...]]
        text = re.sub(r'\[\[([^|\]]+\|)?([^\]]+)\]\]', r'\2', text)
        # remove references <ref>...</ref>
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        # remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # remove URLs
        text = re.sub(r'http\S+', '', text)
        # remove special wiki syntax
        text = re.sub(r"'''|''", '', text)
        text = re.sub(r'={2,}', '', text)
        
        return text.strip()
    
    def extract_words_from_text(self, text: str) -> Set[str]:
        """extract valid georgian words from text."""
        # find all georgian character sequences
        tokens = GEORGIAN_PATTERN.findall(text)
        
        # filter valid words
        valid_words = set()
        for token in tokens:
            # check if it's a valid word (2-20 characters, all georgian)
            if VALID_WORD_PATTERN.match(token) and 2 <= len(token) <= 20:
                valid_words.add(token)
                self.char_freq.update(token)
        
        return valid_words
    
    def process_dump(self, max_articles: int = None, min_words: int = 10000):
        """
        main processing pipeline.
        """
        print("Georgian Wikipedia Processing Pipeline")
        
        # download dump
        if not self.download_wikipedia_dump():
            print("\nUsing fallback method: basic word list")
            self.create_fallback_wordlist()
            return
        
        # extract text
        texts = self.extract_text_from_dump(max_articles=max_articles)
        
        if not texts:
            print("\nNo text extracted, using fallback")
            self.create_fallback_wordlist()
            return
        
        # extract words
        print("\nExtracting Georgian words from articles...")
        for i, text in enumerate(tqdm(texts)):
            words = self.extract_words_from_text(text)
            self.words.update(words)
            
            if (i + 1) % 500 == 0:
                print(f"  Progress: {len(self.words)} unique words found...")
        
        print(f"\nExtraction complete: {len(self.words)} unique words")
        
        # statistics
        self.print_statistics()
        
        # save
        self.save_all()
    
    def create_fallback_wordlist(self):
        """
        create a basic word list if wikipedia download fails.
        uses common georgian words expanded with variations.
        """
        print("\nCreating fallback word list...")
        
        # base common words (expanded list)
        base_words = [
            'გამარჯობა', 'ნახვამდის', 'მადლობა', 'გთხოვთ', 
            'კარგად', 'ცუდად', 'ნორმალურად', 'კეთილი',
            
            'დღეს', 'ხვალ', 'გუშინ', 'დილა', 'დილით', 'საღამო', 'საღამოს',
            'ღამე', 'ღამით', 'დრო', 'დროს', 'საათი', 'საათში', 'წუთი', 'წამი',
            'კვირა', 'ორშაბათი', 'სამშაბათი', 'ოთხშაბათი', 'ხუთშაბათი', 
            'პარასკევი', 'შაბათი',
            'იანვარი', 'თებერვალი', 'მარტი', 'აპრილი', 'მაისი', 'ივნისი',
            'ივლისი', 'აგვისტო', 'სექტემბერი', 'ოქტომბერი', 'ნოემბერი', 'დეკემბერი',
            
            'ერთი', 'ორი', 'სამი', 'ოთხი', 'ხუთი', 'ექვსი', 'შვიდი', 'რვა', 
            'ცხრა', 'ათი', 'თერთმეტი', 'თორმეტი', 'ოცი', 'ოცდაათი', 'ასი', 'ათასი',
            
            'მამა', 'დედა', 'მშობელი', 'მშობლები', 'ძმა', 'და', 'დედა', 'ბიჭი', 
            'გოგო', 'შვილი', 'ქალი', 'კაცი', 'ბავშვი', 'ადამიანი', 'ხალხი', 'ოჯახი',
            'ბებია', 'ბაბუა', 'მამიდა', 'ბიძა',
            
            'წიგნი', 'კალამი', 'რვეული', 'მაგიდა', 'სკამი', 'კარი', 'ფანჯარა',
            'სახლი', 'ოთახი', 'სამზარეულო', 'აბაზანა', 'ეზო', 'ბაღი',
            'ქალაქი', 'სოფელი', 'ქუჩა', 'გზა', 'ქვეყანა', 'მსოფლიო',
            'მდინარე', 'მთა', 'ზღვა', 'ტბა', 'ტყე', 'ველი', 'ციცა', 'ხე',
            'პური', 'წყალი', 'რძე', 'ღვინო', 'ხილი', 'ბოსტნეული', 'ხორცი', 
            'თევზი', 'კვერცხი', 'ყველი', 'პომიდორი', 'კიტრი', 'ვაშლი', 'მსხალი',
            'თხილი', 'ყურძენი', 'კარტოფილი',
            
            'არის', 'იყო', 'იქნება', 'აქვს', 'ჰქონდა', 'ექნება', 'წავა', 'მოვა',
            'აკეთებს', 'აკეთებდა', 'გააკეთა', 'გააკეთებს', 'ამბობს', 'თქვა',
            'ხედავს', 'ნახა', 'უნდა', 'მინდა', 'გინდა', 'შეიძლება',
            'მიდის', 'მივიდა', 'წავიდა', 'მოდის', 'მოვიდა', 'მოვა',
            'კითხულობს', 'წაიკითხა', 'წერს', 'დაწერა', 'ლაპარაკობს',
            
            'კარგი', 'ცუდი', 'დიდი', 'პატარა', 'მცირე', 'დიდებული', 'მაღალი', 
            'დაბალი', 'ახალი', 'ძველი', 'ლამაზი', 'მახინჯი', 'ჭკვიანი', 'სულელი',
            'ძლიერი', 'სუსტი', 'სწრაფი', 'ნელი', 'ცხელი', 'ცივი', 'თბილი',
            'ნათელი', 'ბნელი', 'თეთრი', 'შავი', 'წითელი', 'ლურჯი', 'მწვანე',
            'ყვითელი', 'ნაცრისფერი', 'ნარინჯისფერი',
            
            'საქართველო', 'თბილისი', 'ბათუმი', 'ქუთაისი', 'რუსთავი', 'გორი',
            'ზუგდიდი', 'ფოთი', 'თელავი', 'მცხეთა', 'სიღნაღი', 'ბორჯომი',
            
            'ყველა', 'ყველას', 'არაფერი', 'არავინ', 'არაფრის', 'რაღაც', 'ვიღაც',
            'სადმე', 'როდესაც', 'როცა', 'როგორც', 'თუ', 'რომ', 'და', 'ან', 'მაგრამ',
            'რადგან', 'რატომ', 'როდის', 'როგორ', 'სად', 'ვინ', 'რა', 'რომელი',
            'ეს', 'ის', 'ესენი', 'ისინი', 'ასე', 'ისე', 'აქ', 'იქ', 'ახლა', 'მაშინ',
            
            'კომპიუტერი', 'ტელეფონი', 'ინტერნეტი', 'პროგრამა', 'პროგრამირება',
            'ფაილი', 'დოკუმენტი', 'ელფოსტა', 'საიტი', 'გვერდი',
        ]
        
        self.words = set(base_words)
        
        for word in self.words:
            self.char_freq.update(word)
        
        print(f"Created fallback list with {len(self.words)} words")
        self.print_statistics()
        self.save_all()
    
    def print_statistics(self):
        word_list = sorted(list(self.words))
        lengths = [len(w) for w in word_list]
        
        print("DATASET STATISTICS")
        print(f"Total unique words:      {len(word_list):,}")
        print(f"Character vocabulary:    {len(self.char_freq)}")
        print(f"Shortest word length:    {min(lengths)}")
        print(f"Longest word length:     {max(lengths)}")
        print(f"Average word length:     {sum(lengths)/len(lengths):.1f}")
        print(f"\nMost common characters:  {''.join([c for c, _ in self.char_freq.most_common(20)])}")
        
        print("Sample words (first 30):")
        for i, word in enumerate(word_list[:30], 1):
            print(f"{i:2d}. {word}")
    
    def save_all(self):
        word_list = sorted(list(self.words))
        char_vocab = sorted(list(self.char_freq.keys()))
        
        # save word list
        words_file = self.output_dir / 'georgian_words.txt'
        with open(words_file, 'w', encoding='utf-8') as f:
            for word in word_list:
                f.write(word + '\n')
        print(f"\nSaved word list: {words_file} ({len(word_list)} words)")
        
        # save character vocabulary
        vocab_file = self.output_dir / 'char_vocab.json'
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(char_vocab, f, ensure_ascii=False, indent=2)
        print(f"Saved vocabulary: {vocab_file} ({len(char_vocab)} characters)")
        
        # save metadata
        metadata = {
            'total_words': len(word_list),
            'char_vocab_size': len(char_vocab),
            'char_frequencies': dict(self.char_freq.most_common()),
            'sample_words': word_list[:50]
        }
        metadata_file = self.output_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Saved metadata: {metadata_file}")
        


if __name__ == "__main__":
    processor = GeorgianWikipediaProcessor(output_dir='data')
    
    # set max_articles=5000 for faster processing (~10k-20k words)
    # set max_articles=None for full dump (50k+ words, takes longer)
    processor.process_dump(max_articles=5000, min_words=10000)
    