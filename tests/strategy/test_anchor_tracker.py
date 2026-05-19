"""Tests for the anchor bet result tracker."""


import pytest

from omega.strategy.anchor.tracker import (
    AnchorBetLeg,
    AnchorBetRecord,
    AnchorBetTracker,
)


@pytest.fixture
def tracker(tmp_path):
    """Create a tracker with a temp database."""
    db_path = str(tmp_path / "test_anchor.db")
    t = AnchorBetTracker(db_path=db_path)
    yield t
    t.close()


@pytest.fixture
def sample_bet():
    """A sample bet modeled after the user's OKC @ LAC 4-leg winner."""
    return AnchorBetRecord(
        bet_id="test-okc-lac-001",
        scan_date="2026-04-08",
        game="Thunder @ Clippers",
        legs=[
            AnchorBetLeg(player="Chet Holmgren", team="OKC", stat="pts", threshold=10, hit_rate=1.0, odds_over=-350),
            AnchorBetLeg(player="Shai Gilgeous-Alexander", team="OKC", stat="ast", threshold=5, hit_rate=0.9, odds_over=-200),
            AnchorBetLeg(player="Kris Dunn", team="LAC", stat="ast", threshold=3, hit_rate=0.8, odds_over=-150),
            AnchorBetLeg(player="Brook Lopez", team="LAC", stat="reb", threshold=3, hit_rate=0.9, odds_over=-180),
        ],
        odds_taken=2.20,
        modeled_true_p=0.52,
        result="PENDING",
    )


class TestAnchorBetTracker:
    def test_log_and_retrieve(self, tracker, sample_bet):
        """Log a bet and retrieve it."""
        tracker.log_bet(sample_bet)
        got = tracker.get_bet("test-okc-lac-001")

        assert got is not None
        assert got.bet_id == "test-okc-lac-001"
        assert got.game == "Thunder @ Clippers"
        assert len(got.legs) == 4
        assert got.legs[0].player == "Chet Holmgren"
        assert got.odds_taken == 2.20
        assert got.result == "PENDING"

    def test_idempotent_log(self, tracker, sample_bet):
        """Logging the same bet twice doesn't error."""
        tracker.log_bet(sample_bet)
        tracker.log_bet(sample_bet)
        assert tracker.count() == 1

    def test_grade_bet_win(self, tracker, sample_bet):
        """Grade a bet as WIN and verify CLV computation."""
        tracker.log_bet(sample_bet)
        tracker.grade_bet(
            "test-okc-lac-001",
            result="WIN",
            odds_close=2.05,  # line moved against us
        )

        got = tracker.get_bet("test-okc-lac-001")
        assert got.result == "WIN"
        assert got.odds_close == 2.05
        assert got.clv_pct is not None
        # CLV = (1/2.05 - 1/2.20) * 100 ≈ (0.4878 - 0.4545) * 100 ≈ 3.32%
        assert abs(got.clv_pct - 3.32) < 0.1

    def test_grade_bet_loss(self, tracker, sample_bet):
        """Grade a bet as LOSS."""
        tracker.log_bet(sample_bet)
        tracker.grade_bet("test-okc-lac-001", result="LOSS", notes="Assists leg missed")

        got = tracker.get_bet("test-okc-lac-001")
        assert got.result == "LOSS"
        assert got.notes == "Assists leg missed"

    def test_grade_nonexistent_raises(self, tracker):
        with pytest.raises(ValueError, match="No anchor bet found"):
            tracker.grade_bet("does-not-exist", result="WIN")

    def test_query_by_result(self, tracker, sample_bet):
        """Query filters by result."""
        tracker.log_bet(sample_bet)

        # Create a second bet that's a WIN
        bet2 = AnchorBetRecord(
            bet_id="test-heat-002",
            scan_date="2026-04-09",
            game="Heat vs Raptors",
            legs=[
                AnchorBetLeg(player="Tyler Herro", team="MIA", stat="pts", threshold=15, hit_rate=0.85),
            ],
            odds_taken=2.13,
            result="WIN",
        )
        tracker.log_bet(bet2)

        pending = tracker.query_bets(result="PENDING")
        assert len(pending) == 1
        assert pending[0].bet_id == "test-okc-lac-001"

        wins = tracker.query_bets(result="WIN")
        assert len(wins) == 1
        assert wins[0].bet_id == "test-heat-002"

    def test_summary_stats(self, tracker):
        """Summary stats compute correctly."""
        bets = [
            AnchorBetRecord(
                bet_id="w1", scan_date="2026-04-08", game="Game A",
                legs=[AnchorBetLeg(player="P1", team="T1", stat="pts", threshold=10, hit_rate=0.9)],
                odds_taken=2.20, result="WIN",
            ),
            AnchorBetRecord(
                bet_id="w2", scan_date="2026-04-08", game="Game B",
                legs=[AnchorBetLeg(player="P2", team="T2", stat="pts", threshold=15, hit_rate=0.85)],
                odds_taken=2.10, result="WIN",
            ),
            AnchorBetRecord(
                bet_id="l1", scan_date="2026-04-09", game="Game C",
                legs=[AnchorBetLeg(player="P3", team="T3", stat="ast", threshold=5, hit_rate=0.8)],
                odds_taken=2.00, result="LOSS",
            ),
        ]
        for b in bets:
            tracker.log_bet(b)

        stats = tracker.summary_stats()
        assert stats["graded"] == 3
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert stats["win_rate"] == pytest.approx(2 / 3, abs=0.01)
        # ROI: (2.20 + 2.10 - 3) / 3 * 100 = 43.33%
        assert stats["roi_pct"] == pytest.approx(43.33, abs=0.1)

    def test_export_csv(self, tracker, sample_bet):
        """CSV export produces valid output."""
        tracker.log_bet(sample_bet)
        csv_str = tracker.export_csv()

        assert "bet_id" in csv_str  # header
        assert "test-okc-lac-001" in csv_str
        assert "Chet Holmgren" in csv_str
        lines = csv_str.strip().split("\n")
        assert len(lines) == 2  # header + 1 data row


class TestAnchorBetRecord:
    def test_compute_clv_positive(self):
        """Positive CLV when close odds are shorter."""
        record = AnchorBetRecord(
            bet_id="t1", scan_date="2026-04-08", game="G",
            legs=[], odds_taken=2.20, odds_close=2.05,
        )
        clv = record.compute_clv()
        assert clv is not None
        assert clv > 0  # we got better price than close

    def test_compute_clv_negative(self):
        """Negative CLV when line moved in our favor (we got worse price)."""
        record = AnchorBetRecord(
            bet_id="t2", scan_date="2026-04-08", game="G",
            legs=[], odds_taken=2.00, odds_close=2.30,
        )
        clv = record.compute_clv()
        assert clv is not None
        assert clv < 0

    def test_compute_clv_no_close(self):
        record = AnchorBetRecord(
            bet_id="t3", scan_date="2026-04-08", game="G",
            legs=[], odds_taken=2.00,
        )
        assert record.compute_clv() is None

    def test_generate_id(self):
        id1 = AnchorBetRecord.generate_id()
        id2 = AnchorBetRecord.generate_id()
        assert len(id1) == 12
        assert id1 != id2
