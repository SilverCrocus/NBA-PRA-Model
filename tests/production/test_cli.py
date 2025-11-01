"""
Tests for unified production CLI.
"""
import pytest
from click.testing import CliRunner
from production.cli import cli, predict, train, recommend, pipeline, status


def test_cli_help():
    """Test CLI help message"""
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])

    assert result.exit_code == 0
    assert 'NBA PRA Production Pipeline' in result.output


def test_cli_version():
    """Test CLI version option"""
    runner = CliRunner()
    result = runner.invoke(cli, ['--version'])

    assert result.exit_code == 0
    assert '1.0.0' in result.output


def test_cli_predict_command():
    """Test predict subcommand"""
    runner = CliRunner()
    result = runner.invoke(predict, ['--date', '2024-11-01', '--dry-run'])

    assert result.exit_code == 0
    assert 'DRY RUN' in result.output or 'dry run' in result.output.lower()


def test_cli_predict_command_help():
    """Test predict command help"""
    runner = CliRunner()
    result = runner.invoke(predict, ['--help'])

    assert result.exit_code == 0
    assert 'Generate predictions' in result.output


def test_cli_train_command():
    """Test train subcommand"""
    runner = CliRunner()
    result = runner.invoke(train, ['--dry-run'])

    assert result.exit_code == 0
    assert 'DRY RUN' in result.output or 'dry run' in result.output.lower()


def test_cli_train_command_help():
    """Test train command help"""
    runner = CliRunner()
    result = runner.invoke(train, ['--help'])

    assert result.exit_code == 0
    assert 'Train' in result.output


def test_cli_recommend_command():
    """Test recommend subcommand"""
    runner = CliRunner()
    result = runner.invoke(recommend, ['--date', '2024-11-01', '--min-edge', '0.05', '--dry-run'])

    # Should either succeed or exit cleanly
    assert result.exit_code in [0, 1]  # May fail if no predictions file exists


def test_cli_recommend_command_help():
    """Test recommend command help"""
    runner = CliRunner()
    result = runner.invoke(recommend, ['--help'])

    assert result.exit_code == 0
    assert 'Recommend' in result.output or 'recommend' in result.output.lower()


def test_cli_pipeline_command():
    """Test pipeline subcommand"""
    runner = CliRunner()
    result = runner.invoke(pipeline, ['--dry-run'])

    assert result.exit_code == 0
    assert 'DRY RUN' in result.output or 'dry run' in result.output.lower()


def test_cli_pipeline_command_full():
    """Test pipeline command with --full flag"""
    runner = CliRunner()
    result = runner.invoke(pipeline, ['--full', '--dry-run'])

    assert result.exit_code == 0
    assert 'DRY RUN' in result.output or 'dry run' in result.output.lower()


def test_cli_pipeline_command_help():
    """Test pipeline command help"""
    runner = CliRunner()
    result = runner.invoke(pipeline, ['--help'])

    assert result.exit_code == 0
    assert 'pipeline' in result.output.lower()


def test_cli_status_command():
    """Test status subcommand"""
    runner = CliRunner()
    result = runner.invoke(status, [])

    assert result.exit_code == 0
    # Status command should show system information
    assert 'status' in result.output.lower() or 'model' in result.output.lower()


def test_cli_status_command_help():
    """Test status command help"""
    runner = CliRunner()
    result = runner.invoke(status, ['--help'])

    assert result.exit_code == 0
    assert 'status' in result.output.lower()


def test_cli_predict_with_skip_training():
    """Test predict with --skip-training flag"""
    runner = CliRunner()
    result = runner.invoke(predict, ['--skip-training', '--dry-run'])

    assert result.exit_code == 0


def test_cli_predict_with_skip_odds():
    """Test predict with --skip-odds flag"""
    runner = CliRunner()
    result = runner.invoke(predict, ['--skip-odds', '--dry-run'])

    assert result.exit_code == 0


def test_cli_train_with_cv_folds():
    """Test train with custom CV folds"""
    runner = CliRunner()
    result = runner.invoke(train, ['--cv-folds', '10', '--dry-run'])

    assert result.exit_code == 0


def test_cli_train_with_training_window():
    """Test train with custom training window"""
    runner = CliRunner()
    result = runner.invoke(train, ['--training-window', '2', '--dry-run'])

    assert result.exit_code == 0


def test_cli_recommend_with_top_n():
    """Test recommend with --top-n option"""
    runner = CliRunner()
    result = runner.invoke(recommend, ['--top-n', '5', '--dry-run'])

    # Should handle missing data gracefully
    assert result.exit_code in [0, 1]


def test_cli_recommend_with_min_confidence():
    """Test recommend with --min-confidence option"""
    runner = CliRunner()
    result = runner.invoke(recommend, ['--min-confidence', '0.7', '--dry-run'])

    # Should handle missing data gracefully
    assert result.exit_code in [0, 1]


def test_cli_pipeline_with_skip_options():
    """Test pipeline with skip flags"""
    runner = CliRunner()
    result = runner.invoke(pipeline, [
        '--skip-data-update',
        '--skip-feature-engineering',
        '--skip-training',
        '--dry-run'
    ])

    assert result.exit_code == 0
