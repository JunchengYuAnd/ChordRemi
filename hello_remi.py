"""
简单的自定义REMI tokenizer，添加和弦标记功能
"""

from pathlib import Path
from typing import Union, Optional
from symusic import Score
from miditok import REMI, TokenizerConfig
from miditok.classes import Event, TokSequence
from miditoolkit.midi import parser as midi_parser
import tempfile
import os
from chorder import Dechorder

class ChordREMI(REMI):
    """
    自定义REMI tokenizer，添加和弦分析结果作为tokens
    
    这个类演示了如何扩展REMI tokenizer并添加和弦标记
    """
    
    def __init__(
        self,
        tokenizer_config: TokenizerConfig = None,
        max_bar_embedding: Optional[int] = None,
        params: Optional[Union[str, Path]] = None,
    ):
        # 创建自定义token类型前，先确保配置正确
        if tokenizer_config is not None:
            # 确保additional_params存在
            if tokenizer_config.additional_params is None:
                tokenizer_config.additional_params = {}
            
            # 标记我们将添加Chord token
            tokenizer_config.additional_params["use_chord_tokens"] = True
            
        # 调用父类初始化
        super().__init__(tokenizer_config, max_bar_embedding, params)
        
        # 存储已检测到的和弦信息，按轨道和时间索引
        self.detected_chords = {}
    
    def _tweak_config_before_creating_voc(self) -> None:
        """在创建词汇表前调整配置（这是miditok中正确的扩展点）"""
        # 首先调用父类方法
        super()._tweak_config_before_creating_voc()
            
        # 添加use_chord_tokens标志
        if "use_chord_tokens" not in self.config.additional_params:
            self.config.additional_params["use_chord_tokens"] = True
    
    def _create_base_vocabulary(self) -> list[str]:
        """创建基础词汇表，添加和弦标记"""
        # 首先调用父类方法获取基础词汇表
        vocab = super()._create_base_vocabulary()
        
        # 添加Chord相关tokens
        # 根音（C, C#, D, ..., B）
        root_notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        # 常见和弦类型
        chord_qualities = ["maj", "min", "dim", "aug"]
        
        # 将根音和和弦类型组合成完整的和弦token
        for root in root_notes:
            for quality in chord_qualities:
                vocab.append(f"Chord_{root}_{quality}")
        
        # 添加NoChord token用于无和弦区域
        vocab.append("Chord_NoChord")
        
        #print(f"[DEBUG] 添加了 {len(root_notes) * len(chord_qualities) + 1} 个Chord tokens")
        
        return vocab
    
    def _create_token_types_graph(self) -> dict[str, set[str]]:
        """创建标记类型图，定义Chord标记的转换关系"""
        # 获取父类的转换图
        graph = super()._create_token_types_graph()
        
        # 添加Chord的转换规则
        # 1. Bar后面可以跟Chord
        if "Bar" in graph:
            graph["Bar"].add("Chord")
            
        # 2. Position后面可以跟Chord（用于小节中间的和弦变化）
        if "Position" in graph:
            graph["Position"].add("Chord")
        
        # 3. Chord后面可以跟什么（与Position类似）
        graph["Chord"] = set()
        if "Position" in graph:
            for token_type in graph["Position"]:
                graph["Chord"].add(token_type)
        
        #print(f"[DEBUG] token类型图: Chord可以跟{graph['Chord']}")
        #print(f"[DEBUG] token类型图: Position可以跟Chord")
        
        return graph
    
    def _add_time_events(self, events: list[Event], time_division: int) -> list[Event]:
        """
        添加时间相关事件（包括Chord事件）
        
        Args:
            events: 事件列表
            time_division: 每拍的tick数
            
        Returns:
            添加了时间事件的事件列表
        """
        # 首先获取父类添加的时间事件
        time_events = super()._add_time_events(events, time_division)
        
        # 然后添加Chord事件（如果已启用）
        all_events = []
        current_track_idx = 0  # 默认使用轨道0，而不是None
        current_bar_idx = 0  # 跟踪实际小节索引，不使用Bar的value
        
        # 记录已添加的和弦事件数量
        chord_events_added = 0
        
        # 检查是否启用了和弦token
        use_chord_tokens = self.config.additional_params.get("use_chord_tokens", False)
        if use_chord_tokens:
            #print(f"[DEBUG] 和弦token功能已启用")
            pass
        else:
            #print(f"[DEBUG] 和弦token功能未启用，将不添加和弦事件")
            return time_events  # 如果未启用和弦token，直接返回原始事件
        
        # 根音数字到字母的映射
        pitch_class_to_name = {
            0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F",
            6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"
        }
        
        # 调试：检查detected_chords内容
        if self.detected_chords:
            #print(f"[DEBUG] 和弦检测结果存在，包含 {len(self.detected_chords)} 个轨道")
            for track_idx, chords in self.detected_chords.items():
                #print(f"[DEBUG] 轨道 {track_idx} 有 {len(chords)} 个和弦")
                if chords:
                    sample_chord = chords[0]
                    #print(f"[DEBUG] 示例和弦: {sample_chord}")
        else:
            #print("[DEBUG] 警告: 没有和弦检测结果!")
            pass
        
        # 输出所有事件类型
        event_types = set(e.type_ for e in time_events)
        #print(f"[DEBUG] 事件类型: {event_types}")
        
        # 没有Track事件时使用默认轨道
        has_track_event = any(e.type_ == "Track" for e in time_events)
        if not has_track_event:
            #print(f"[DEBUG] 没有找到Track事件，使用默认轨道: {current_track_idx}")
            pass
        
        for event in time_events:
            # 如果是Track事件，记录当前轨道
            if event.type_ == "Track":
                current_track_idx = event.value
                #print(f"[DEBUG] 遇到Track事件，当前轨道ID: {current_track_idx}")
                current_bar_idx = 0  # 重置小节计数
            
            # 如果是Bar事件，增加小节计数，并添加两个半小节的和弦
            if event.type_ == "Bar":
                #print(f"[DEBUG] 遇到Bar事件: {event.type_}_{event.value}, 小节索引: {current_bar_idx}, 当前轨道: {current_track_idx}")
                all_events.append(event)
                
                if (self.detected_chords and 
                    current_track_idx in self.detected_chords):
                    
                    # 1. 添加第一个半小节的和弦
                    #print(f"[DEBUG] 轨道 {current_track_idx} 有和弦数据，查找小节 {current_bar_idx} 的第一个半小节和弦")
                    chord_info_1 = None
                    for chord_data in self.detected_chords[current_track_idx]:
                        measure_idx, half_in_measure, root_note, chord_quality, _ = chord_data
                        if measure_idx == current_bar_idx and half_in_measure == 1:
                            chord_info_1 = (root_note, chord_quality)
                            #print(f"[DEBUG] 找到小节 {current_bar_idx} 的第一个半小节和弦: 根音 {root_note}, 和弦类型 {chord_quality}")
                            break
                    
                    if chord_info_1 is None:
                        #print(f"[DEBUG] 没有找到小节 {current_bar_idx} 的第一个半小节和弦")
                        pass
                    
                    # 添加第一个半小节的和弦事件
                    chord_event_1 = self._create_chord_event(chord_info_1, event.time, event.desc, pitch_class_to_name)
                    if chord_event_1:
                        all_events.append(chord_event_1)
                        chord_events_added += 1
                    
                    # 2. 添加第二个半小节的和弦
                    #print(f"[DEBUG] 轨道 {current_track_idx} 有和弦数据，查找小节 {current_bar_idx} 的第二个半小节和弦")
                    chord_info_2 = None
                    for chord_data in self.detected_chords[current_track_idx]:
                        measure_idx, half_in_measure, root_note, chord_quality, _ = chord_data
                        if measure_idx == current_bar_idx and half_in_measure == 2:
                            chord_info_2 = (root_note, chord_quality)
                            #print(f"[DEBUG] 找到小节 {current_bar_idx} 的第二个半小节和弦: 根音 {root_note}, 和弦类型 {chord_quality}")
                            break
                    
                    # 处理第二个半小节的和弦
                    if chord_info_2 is None:
                       # print(f"[DEBUG] 没有找到小节 {current_bar_idx} 的第二个半小节和弦")
                        # 如果找不到第二个半小节的和弦，使用第一个半小节的和弦
                        chord_info_2 = chord_info_1
                        if chord_info_2:
                            #print(f"[DEBUG] 使用第一个半小节的和弦作为第二个半小节的和弦")
                            pass
                    else:
                        # 检查第二个半小节的和弦是否为"无和弦"，而第一个半小节有有效和弦
                        if chord_info_2[0] == "无和弦" and chord_info_1 is not None and chord_info_1[0] != "无和弦" and chord_info_1[0] != "":
                            chord_info_2 = chord_info_1
                            #print(f"[DEBUG] 第二个半小节的和弦是无和弦，而第一个半小节有有效和弦，使用第一个半小节的和弦")
                    
                    # 添加第二个半小节的和弦事件
                    chord_event_2 = self._create_chord_event(chord_info_2, event.time, event.desc, pitch_class_to_name)
                    if chord_event_2:
                        all_events.append(chord_event_2)
                        chord_events_added += 1
                else:
                    if not self.detected_chords:
                        #print("[DEBUG] 没有和弦检测结果")
                        pass
                    elif current_track_idx not in self.detected_chords:
                        #print(f"[DEBUG] 轨道 {current_track_idx} 没有和弦数据")
                        pass
                
                current_bar_idx += 1  # 增加小节计数
            else:
                all_events.append(event)
        
        # 添加调试信息
        #print(f"[DEBUG] _add_time_events: 添加了 {chord_events_added} 个和弦事件")
        
        return all_events
    
    def _create_chord_event(self, chord_info, time, desc, pitch_class_to_name):
        """
        创建和弦事件
        
        Args:
            chord_info: 和弦信息元组 (root_note, chord_quality)
            time: 事件时间
            desc: 事件描述
            pitch_class_to_name: 音高类别到音名的映射字典
            
        Returns:
            创建的和弦事件，如果无法创建则返回None
        """
        if not chord_info:
            return None
            
        root, quality = chord_info
        if root != "无和弦" and root != "":
            # 如果根音是数字，转换为字母
            if isinstance(root, int) or (isinstance(root, str) and root.isdigit()):
                root_idx = int(root) % 12
                root = pitch_class_to_name.get(root_idx, "C")
                #print(f"[DEBUG] 将数字根音 {chord_info[0]} 转换为 {root}")
                
            # 转换和弦质量为标准名称
            if quality == "m":
                quality = "min"  # 转换为标准名称
            elif quality == "M":
                quality = "maj"  # 大三和弦，M转换为maj
            elif quality == "dim":
                quality = "dim"  # 保持不变
            elif quality == "aug":
                quality = "aug"  # 保持不变
            # 其他和弦类型根据需要添加更多转换规则
            
            # 确保和弦质量在词汇表中
            chord_token_test = f"Chord_{root}_{quality}"
            if chord_token_test in self.vocab:
                chord_token = chord_token_test
                #print(f"[DEBUG] 转换后的和弦token: {chord_token} (在词汇表中)")
            else:
                # 尝试找到最接近的和弦质量
                found = False
                for q in ["maj", "min", "dim", "aug"]:
                    test_token = f"Chord_{root}_{q}"
                    if test_token in self.vocab:
                        chord_token = test_token
                        #print(f"[DEBUG] 找到替代和弦token: {chord_token}")
                        found = True
                        break
                
                if not found:
                    chord_token = "Chord_NoChord"
                    #print(f"[DEBUG] 找不到匹配的和弦token, 使用NoChord")
        else:
            chord_token = "Chord_NoChord"
        
        #print(f"[DEBUG] 添加和弦: 和弦:{chord_token}")
        
        chord_event = Event(
            type_="Chord",
            value=chord_token.replace("Chord_", ""),  # 保存不带前缀的值
            time=time,
            desc=desc
        )
        return chord_event
    
    def _track_to_tokens(self, track: list[Event], time_division: int) -> list[str]:
        """
        将track事件转换为tokens，确保加入和弦token
        
        Args:
            track: 轨道事件列表
            time_division: 每拍的tick数
            
        Returns:
            token序列
        """
        # 调试：检查track中是否有Chord事件
        chord_events = [e for e in track if e.type_ == "Chord"]
        print(f"[DEBUG] _track_to_tokens: 输入的track包含 {len(chord_events)} 个Chord事件")
        if chord_events:
            print(f"[DEBUG] 前3个Chord事件: {chord_events[:3]}")
        
        # 手动处理token生成
        tokens = []
        for event in track:
            token = None
            
            # 处理特殊事件类型
            if event.type_ == "Chord":
                # 创建和弦token
                chord_token = f"Chord_{event.value}"
                
                # 检查token是否在词汇表中
                if chord_token in self.vocab:
                    token = chord_token
                    print(f"[DEBUG] 使用和弦token: {token}")
                else:
                    # 如果不在词汇表中，尝试修复"M"到"maj"的问题
                    if "_M" in chord_token:
                        fixed_token = chord_token.replace("_M", "_maj")
                        if fixed_token in self.vocab:
                            token = fixed_token
                            print(f"[DEBUG] 修复和弦token: {chord_token} -> {token}")
                        else:
                            # 如果仍然不在词汇表中，使用NoChord
                            token = "Chord_NoChord"
                            print(f"[DEBUG] 无法处理和弦token {chord_token}，使用NoChord")
                    else:
                        # 其他未知情况，使用NoChord
                        token = "Chord_NoChord"
                        print(f"[DEBUG] 无法识别的和弦token: {chord_token}，使用NoChord")
            
            # 如果不是特殊事件，使用父类方法处理
            if token is None:
                if event.type_ in self.vocab_types_idx:
                    # 这是标准事件类型，使用标准方法处理
                    token = self._event_to_token(event)
                else:
                    # 跳过未知事件类型
                    continue
            
            tokens.append(token)
        
        # 调试：检查tokens中是否有Chord token
        chord_tokens = [t for t in tokens if t.startswith("Chord_")]
        print(f"[DEBUG] _track_to_tokens: 输出的tokens包含 {len(chord_tokens)} 个Chord token")
        if chord_tokens:
            print(f"[DEBUG] Chord tokens: {chord_tokens[:5]}")
        
        return tokens
    
    def encode(self, midi_obj: Union[str, Path], *args, **kwargs):
        """
        重写encode方法，在tokenize前先进行和弦检测
        
        Args:
            midi_obj: MIDI对象或文件路径
            
        Returns:
            tokenized序列
        """
        # 检查是否启用了和弦token
        use_chord_tokens = self.config.additional_params.get("use_chord_tokens", False)
        
        # 首先加载MIDI文件（如果必要）
        if isinstance(midi_obj, (str, Path)):
            midi_data = Score(midi_obj)
        else:
            midi_data = midi_obj
            
        # 进行和弦检测，并存储结果（如果启用了和弦token）
        self.detected_chords = {}
        
        if use_chord_tokens:
            #print(f"[DEBUG] 和弦token功能已启用，开始检测和弦")
            #print(f"[DEBUG] 开始为 {len(midi_data.tracks)} 个轨道检测和弦")
            
            for track_idx, track in enumerate(midi_data.tracks):
                if len(track.notes) == 0:
                    #print(f"[DEBUG] 轨道 {track_idx} 没有音符，跳过")
                    continue
                    
                #print(f"[DEBUG] 正在检测轨道 {track_idx} 的和弦 ({len(track.notes)} 个音符)")
                # 检测当前轨道的和弦
                track_chords = self.detect_chords_for_track(midi_data, track_idx)
                
                if track_chords:
                    self.detected_chords[track_idx] = track_chords
                    #print(f"[DEBUG] 轨道 {track_idx} 检测到 {len(track_chords)} 个和弦")
                else:
                    #print(f"[DEBUG] 轨道 {track_idx} 没有检测到和弦")
                    pass
                    
            #print(f"[DEBUG] 和弦检测完成，共 {len(self.detected_chords)} 个轨道有和弦信息")
        else:
            print(f"[DEBUG] 和弦token功能未启用，跳过和弦检测")
        
        # 调用父类的encode方法进行实际的tokenization
        return super().encode(midi_data, *args, **kwargs)
    
    def detect_chords_for_track(self, midi_data: Union[str, Path], track_idx: int) -> list:
        """
        对特定轨道进行和弦检测
        
        Args:
            midi_data: symusic.Score对象
            track_idx: 要检测的轨道索引
            
        Returns:
            检测到的和弦列表
        """
        # 默认 4/4 拍
        time_sig_numerator = 4
        time_sig_denominator = 4

        # 从 symusic Score 中获取时间签名
        if len(midi_data.time_signatures) > 0:
            # 获取第一个时间签名
            time_sig = midi_data.time_signatures[0]
            time_sig_numerator = time_sig.numerator
            time_sig_denominator = time_sig.denominator
        
        # 计算每小节的拍数（例如 4/4 拍为 4 拍）
        beats_per_measure = time_sig_numerator * (4 / time_sig_denominator)
        # 计算半小节的拍数
        half_measure_beats = int(beats_per_measure / 2)
        
        # 创建仅包含指定轨道的临时MIDI
        temp_midi = Score()
        # 设置时间分割值
        temp_midi.ticks_per_quarter = midi_data.ticks_per_quarter
        temp_midi.time_signatures = midi_data.time_signatures
        temp_midi.tempos = midi_data.tempos
        
        # 只添加指定轨道
        if track_idx < len(midi_data.tracks):
            temp_midi.tracks = [midi_data.tracks[track_idx]]
        else:
            return []  # 轨道不存在

        # 创建临时MIDI文件
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_file:
            temp_midi_path = temp_file.name
        
        try:
            # 将临时Score对象保存为MIDI文件
            temp_midi.dump_midi(temp_midi_path)
            
            # 使用miditoolkit加载临时MIDI文件
            midi_obj = midi_parser.MidiFile(temp_midi_path)
            
            # 以每半小节为单位进行和弦检测
            chords_per_half = Dechorder.get_chords(midi_obj, beat=half_measure_beats)
            
            # 组织输出
            half_measure_chords = []
            for idx, (chord, score_value) in enumerate(chords_per_half):
                measure_idx = idx // 2      # 每小节2个半小节
                half_in_measure = (idx % 2) + 1
                if chord.is_complete():
                    root_note = chord.root_pc   # 根音
                    chord_quality = chord.quality  # 和弦性质
                    half_measure_chords.append((measure_idx, half_in_measure, root_note, chord_quality, score_value))
                else:
                    half_measure_chords.append((measure_idx, half_in_measure, "无和弦", "", score_value))
            
            return half_measure_chords
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_midi_path):
                os.remove(temp_midi_path)
    
    def detect_chords(self, midi_data: Union[str, Path]) -> list:
        """
        直接对MIDI数据进行和弦检测，不经过tokenization流程
        
        Args:
            midi_data: symusic.Score对象
            
        Returns:
            检测到的和弦列表
        """
        # 默认 4/4 拍
        time_sig_numerator = 4
        time_sig_denominator = 4

        # 从 symusic Score 中获取时间签名
        if len(midi_data.time_signatures) > 0:
            # 获取第一个时间签名
            time_sig = midi_data.time_signatures[0]
            time_sig_numerator = time_sig.numerator
            time_sig_denominator = time_sig.denominator
        
        # 计算每小节的拍数（例如 4/4 拍为 4 拍）
        beats_per_measure = time_sig_numerator * (4 / time_sig_denominator)
        # 计算半小节的拍数
        half_measure_beats = int(beats_per_measure / 2)

        # 由于 Dechorder 可能不直接支持 symusic.Score 对象，
        # 需要创建临时 MIDI 文件作为中间转换步骤
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_file:
            temp_midi_path = temp_file.name
        
        try:
            # 将 symusic.Score 对象保存为临时 MIDI 文件
            midi_data.dump_midi(temp_midi_path)
            
            # 使用 miditoolkit 加载临时 MIDI 文件供 Dechorder 使用
            from miditoolkit.midi import parser as midi_parser
            midi_obj = midi_parser.MidiFile(temp_midi_path)
            
            # 以每半小节为单位进行和弦检测
            chords_per_half = Dechorder.get_chords(midi_obj, beat=half_measure_beats)
            
            # 组织输出，每个条目表示一个半小节的和弦结果
            half_measure_chords = []
            for idx, (chord, score_value) in enumerate(chords_per_half):
                measure_idx = idx // 2      # 每小节2个半小节
                half_in_measure = (idx % 2) + 1
                if chord.is_complete():
                    root_note = chord.root_pc   # 根音
                    chord_quality = chord.quality  # 和弦性质
                    half_measure_chords.append((measure_idx, half_in_measure, root_note, chord_quality, score_value))
                else:
                    half_measure_chords.append((measure_idx, half_in_measure, "无和弦", "", score_value))
            
            return half_measure_chords
        finally:
            # 清理临时文件
            if os.path.exists(temp_midi_path):
                os.remove(temp_midi_path)

    def _token_to_event(self, token: str) -> Event:
        """
        将token转换为事件，处理Chord tokens的特殊情况
        
        Args:
            token: 要转换的token
            
        Returns:
            对应的Event对象
        """
        # 处理和弦token的特殊情况
        if token.startswith("Chord_"):
            # 和弦token格式为 Chord_Root_Quality
            # 由于和弦token中可能有多个下划线，需要特殊处理
            chord_parts = token.split("_", 1)  # 只在第一个下划线处分割
            type_ = chord_parts[0]
            value = chord_parts[1] if len(chord_parts) > 1 else ""
            
            return Event(
                type_=type_,
                value=value,
                time=None,
                desc=None
            )
        
        # 对于非和弦token，使用父类的方法
        return super()._token_to_event(token)
    
    def _tokens_to_score(
        self,
        tokens: TokSequence | list[TokSequence],
        programs: list[tuple[int, bool]] | None = None,
    ) -> TokSequence:
        """
        重写父类的方法以正确处理Chord tokens
        
        Args:
            tokens: 要转换的tokens
            programs: 轨道的乐器程序
            
        Returns:
            转换后的Score对象
        """
        # 预处理tokens，处理Chord token
        processed_tokens = []
        
        if isinstance(tokens, TokSequence):
            # 单个序列
            new_tokens = []
            for token in tokens.tokens:
                # 特殊处理Chord tokens以防止split时出错
                if token.startswith("Chord_"):
                    # 安全地跳过Chord tokens，因为它们在MIDI生成时不需要
                    continue
                else:
                    new_tokens.append(token)
            
            # 创建新的TokSequence
            new_tok_seq = TokSequence(tokens=new_tokens)
            new_tok_seq.ids = tokens.ids  # 保留原始的ids
            processed_tokens = new_tok_seq
        else:
            # 多个序列
            processed_tokens = []
            for seq in tokens:
                new_tokens = []
                for token in seq.tokens:
                    # 特殊处理Chord tokens
                    if token.startswith("Chord_"):
                        # 安全地跳过Chord tokens
                        continue
                    else:
                        new_tokens.append(token)
                
                # 创建新的TokSequence
                new_tok_seq = TokSequence(tokens=new_tokens)
                new_tok_seq.ids = seq.ids  # 保留原始的ids
                processed_tokens.append(new_tok_seq)
        
        # 调用父类方法处理处理后的tokens
        return super()._tokens_to_score(processed_tokens, programs)

if __name__ == "__main__":
    import argparse
    
    # 定义参数解析器
    parser = argparse.ArgumentParser(description="测试ChordREMI tokenizer")
    parser.add_argument("--input", "-i", help="输入MIDI文件路径", default="exp_valid_02\processed_0BN8QGuy_0#d-2.mid")
    parser.add_argument("--output", "-o", help="输出MIDI文件路径", default="output.mid")
    parser.add_argument("--verbose", "-v", action="store_true", help="启用详细输出")
    
    args = parser.parse_args()
    
    # 创建tokenizer配置
    TOKENIZER_PARAMS = {
        "num_velocities": 32,
        "special_tokens": ["PAD", "BOS", "EOS"],
        "use_tempos": True,
        "use_programs": False,  # 单轨道
        "num_tempos": 32,
        "tempo_range": (50, 200),
        # 配置额外参数
        "additional_params": {
            "use_chord_tokens": True  # 确保配置中包含和弦token
        }
    }
    
    try:
        # 创建配置对象
        config = TokenizerConfig(**TOKENIZER_PARAMS)
        
        # 创建自定义tokenizer
        tokenizer = ChordREMI(config)
        
        # 打印tokenizer的信息
        print(f"Tokenizer vocabulary size: {len(tokenizer)}")
        
        # 检查输入文件是否存在
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} does not exist.")
            exit(1)
            
        # 读取MIDI文件
        print(f"Reading MIDI file: {args.input}")
        input_midi = Score(args.input)

        chord_results = tokenizer.detect_chords(input_midi)

        def print_half_measure_chords(half_measure_chords):
            """打印按半小节划分的和弦检测结果，分开显示根音和和弦性质"""
            print("\n=== 按半小节的和弦分析结果 ===")
            for measure_idx, half_in_measure, root_note, chord_quality, score in half_measure_chords:
                if root_note == "无和弦":
                    print(f"第 {measure_idx+1} 小节，第 {half_in_measure} 个半小节: 无和弦 (score: {score})")
                else:
                    print(f"第 {measure_idx+1} 小节，第 {half_in_measure} 个半小节: 根音: {root_note} 和弦性质: {chord_quality} (score: {score})")

        print_half_measure_chords(chord_results)
                
        # 输出MIDI文件的基本信息
        print(f"MIDI file loaded: {len(input_midi.tracks)} tracks, {input_midi.ticks_per_quarter} ticks per beat")
        
        # 标记化
        print("Tokenizing MIDI...")
        input_tokens = tokenizer(input_midi)
        print(input_tokens[0].tokens)
        
        # 打印token信息
        for i, seq in enumerate(input_tokens):
            print(f"\nTrack {i} tokens (前30个):")
            print(seq.tokens[:100])
            
            # 计算Bar和Chord标记的数量
            bar_tokens = [t for t in seq.tokens if t.startswith("Bar_")]
            # 检查和弦token的数量
            chord_tokens = [t for t in seq.tokens if t.startswith("Chord_")]
            
            print(f"Bar tokens: {len(bar_tokens)}")
            print(f"Chord tokens: {len(chord_tokens)}")
            
            if len(chord_tokens) > 0:
                print(f"First 5 chord tokens: {chord_tokens[:5]}")
            else:
                print("没有找到任何和弦token!")
                
        
        # 解码回MIDI
        print("\n解码tokens回MIDI...")
        try:
            output_midi = tokenizer.decode(input_tokens)
            
            # 保存结果
            output_midi.dump_midi(args.output)
            print(f"Output saved to {args.output}")
        except Exception as e:
            print(f"解码过程出错: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
        
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc() 